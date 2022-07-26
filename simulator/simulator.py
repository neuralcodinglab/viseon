if __name__ == '__main__':
    import sys
    from os.path import dirname, abspath                     
    sys.path.insert(0, dirname(dirname(abspath(__file__)))) 

import torch
import math

import simulator.utils as utils
import simulator.init as init

class GaussianSimulator(object):
    def __init__(self, params, r, phi, batch_size=1, device='cpu'):
        """initialize a simulator with provided parameters settings, given phosphene locations in polar coordinates

        :param params: dict of dicts with all setting parameters
        :param r: eccentricities of phosphenes
        :param phi: angles of phosphenes
        """
        
        # Settings
        # self.params = params
        # # Other parameters
        self.__dict__.update(params)

        #gpu_nr = self.run['gpu']
        self.device = device
        self.batch_size = batch_size

        # Convert to tensors, generate pMaps
        r = torch.from_numpy(r).to(self.device).float()
        phi = torch.from_numpy(phi).to(self.device).float()
        
        # Useful variables
        self.deg2pix_coeff = utils.get_deg2pix_coeff(self.run)
        self.resolution = self.run['resolution']
        self.n_phosphenes = len(r)

        # Generate pMaps
        self.pMap, self.invalid_electrodes = self.generate_pMaps(r, phi) 

        # Init phosphene properties
        self.magnification = init.init_magnification(params, r)
        # self.magnification = torch.from_numpy(self.magnification).to(self.device)
        # assert self.magnification.device == self.device, f"Device not correct, self.magnification is on '{self.magnification.device}' while simulator device is '{self.device}'"

        if self.thresholding['use_threshold']:
            # threshold, slope_thresh = init.init_threshold_curves(self.n_phosphenes, self.thresholding['range_thresh'], self.thresholding['range_slope'])
            threshold_mean = self.thresholding['rheobase']*self.thresholding['chronaxie']
            threshold, slope_thresh = init.init_threshold_curves(self.n_phosphenes, threshold_mean, self.thresholding['threshold_sd'], self.thresholding['range_slope'])
            self.threshold = torch.from_numpy(threshold).to(self.device).float()
            self.slope_thresh =  torch.from_numpy(slope_thresh).to(self.device).float()

        # Temporal dynamics params
        self.activation_decay = math.exp(-self.temporal_dynamics['activation_decay_rate']*(1/self.run['fps']))
        self.trace_fast_decay = math.exp(-self.temporal_dynamics['trace_fast_decay_rate']*(1/self.run['fps']))
        self.trace_slow_decay = math.exp(-self.temporal_dynamics['trace_slow_decay_rate']*(1/self.run['fps']))
        self.trace_fast_increase = self.temporal_dynamics['trace_fast_increase_rate']*(1/self.run['fps'])
        self.trace_slow_increase = self.temporal_dynamics['trace_slow_increase_rate']*(1/self.run['fps'])

        # Sigmoid for brightness saturation & thresholding psychometric curves
        self.SIGMOID = lambda x: 1 / (1 + torch.exp(-x))

        # Set activation mask for sampling
        receptive_field_size = params['sampling']['RF_size']
        self.sampling_mask = (self.pMap.permute(1,2,0) < (self.deg2pix_coeff*(receptive_field_size*(1/self.magnification)))).permute(2,0,1).float()

        # Reset memory 
        self.reset()  
    
    def reset(self):
        """Reset Memory of previous timestep"""
        self.sigma = torch.zeros(self.batch_size,len(self.pMap)).to(self.device)
        self.activation = torch.zeros(self.batch_size,len(self.pMap)).to(self.device)

        if self.run['video']:
            self.trace_slow = torch.zeros(self.batch_size,len(self.pMap)).to(self.device)
            self.trace_fast = torch.zeros(self.batch_size,len(self.pMap)).to(self.device)
    
    def generate_pMaps(self, r, phi):
        """generate phosphene maps (for each phosphene distance to each pixel)

        :param r: tensor of eccentricities of phosphenes
        :param phi: tensor of angles of phosphenes
        :return: an (n_phosphenes x resolution[0] x resolution[1]) array describing distances from phosphene locations
        """
        N_PHOSPHENES = len(r)
        GABOR_ORIENTATION = lambda size: torch.rand(size)*2*math.pi #Rotation of ellips

        # Cartesian coordinate system for the visual field
        x = torch.arange(self.resolution[0], device=self.device).float()
        y = torch.arange(self.resolution[1], device=self.device).float()
        grid = torch.meshgrid(x, y, indexing='xy')
        grid_x = grid[0].unsqueeze(0).repeat(N_PHOSPHENES, 1, 1)
        grid_y = grid[1].unsqueeze(0).repeat(N_PHOSPHENES, 1, 1)
        grid = (grid_x,grid_y)

        # Phosphene mapping (radial distances to center of phosphene) 
        pMap = torch.zeros(len(r), *self.resolution, device= self.device)

        # Eccentricities in pixels
        ecc = self.deg2pix_coeff*r 

        # Convert to cartesian indices
        x_offset = torch.round(torch.cos(phi)*ecc) + self.resolution[0]//2
        y_offset = torch.round(torch.sin(phi)*ecc) + self.resolution[1]//2

        # Check if phosphene locations are inside of view angle
        invalid_electrodes = ((x_offset <=0) | (x_offset >= self.resolution[0])) | ((y_offset <= 0) | (y_offset >= self.resolution[1]))
        print(f"{torch.sum(invalid_electrodes)} phosphenes are outside of view, will not be shown")

        # x_offset = torch.clamp(x_offset.int(),0,self.resolution[0]) #DISCUSS: I removed this, since we don't want to show them. Is that okay?
        # y_offset = torch.clamp(y_offset.int(),0,self.resolution[1])
        
        x = grid[0]-x_offset.view(-1,1,1)
        y = grid[1]-y_offset.view(-1,1,1)

        if self.gabor['gabor_filtering']:
            theta = GABOR_ORIENTATION(N_PHOSPHENES).view(-1,1,1)#np.expand_dims(GABOR_ORIENTATION(N_PHOSPHENES),(1,2))
            y_rotated = (-x * torch.sin(theta)) + (y * torch.cos(theta))
            x_rotated = (x * torch.cos(theta)) + (y * torch.sin(theta))
            pMap = torch.sqrt(x_rotated**2 + (self.gabor['gamma']**2)*y_rotated**2)
        else:
            pMap = torch.sqrt(x**2 + y**2)

        return pMap, invalid_electrodes
    
    def update(self, stim_amp, pw=None, freq=None): 
        """Adjust state as function of previous state and current stimulation. 
        NOTE: all parameters (such as intensity_decay) must be provided at initialization"""
        # DISCUSS: check order of transformations

        # if stim is only stimulation amplitude, add other parameters from default:
        if pw is None:
            pw = self.default_stim['pw_default']*torch.ones_like(stim_amp)
        if freq is None:
            freq = self.default_stim['freq_default']*torch.ones_like(stim_amp)

        stim = torch.stack((stim_amp,pw,freq),dim=2) #TODO: check!
        
        #calculate charge per second
        charge_per_second = torch.prod(stim, dim=2) #in C
        cps_inefficiency = self.thresholding['rheobase']*pw*freq
        effective_cps = torch.max(charge_per_second - cps_inefficiency, torch.zeros_like(charge_per_second))
        
        if self.run['video']:
            # update activation using charge per frame and trace for habituation 
            self.activation = self.activation_decay * self.activation + self.temporal_dynamics['input_effect'] * (effective_cps - (self.trace_fast + self.trace_slow))  
            self.activation = torch.max(self.activation, torch.zeros_like(self.activation))  # don't allow negative activations
            self.trace_fast = self.trace_fast_decay * self.trace_fast + self.trace_fast_increase * effective_cps  # update fast trace
            self.trace_slow = self.trace_slow_decay * self.trace_slow + self.trace_slow_increase * effective_cps  # update slow trace
        else:
            self.activation = effective_cps
        # Bosking et al., 2017: input current effect on size, including saturation at higher values
        # AC = self.MD / torch.add(torch.exp(-self.slope_stim*(stim-self.I_half)),1) # DISCUSS: activation or stimulus amplitude here? No temporal dynamics now

        # or Tehovnik 2007? 
        AC = (stim[:,:,0]/self.size_saturation['current_spread'])**0.5

        self.sigma = self.deg2pix_coeff*(AC*1/self.magnification.repeat(self.batch_size,1)) #+ self.activation_decay*self.sigma #TODO: check these temporal dynamics

        # saturate activation values
        self.brightness = self.SIGMOID(self.brightness_saturation['slope_brightness']*(self.activation-self.brightness_saturation['cps_half'])).float()
        # self.brightness = torch.tanh(self.brightness_saturation['slope_brightness']*self.activation) #DISCUSS: sigmoid or tanh?

        # thresholding. Determine whether the activation is high enough (probabilistically) and apply to brightness vector
        if self.thresholding['use_threshold']: #TODO: check
            probs = self.SIGMOID(self.slope_thresh*(self.activation-self.threshold))
            spikes = torch.rand(probs.shape, device=self.device) < probs
            if self.run['print_stats']:
                print(f"{torch.sum(spikes)} out of {len(spikes)} cross threshold")
            # spikes = self.activation > self.threshold # to test non-probabilistic threshold
            self.brightness = spikes*self.brightness  # TODO: effect of threshold over time: multiply here with activation, or with copy of activation (not sure how this works combined with the trace)

        if self.run['print_stats']: 
            self.print_stats('charge per second:', charge_per_second)
            self.print_stats('sigma (in pixels)', self.sigma)
            self.print_stats('activation', self.activation)
            self.print_stats('sigmoided activation', self.brightness)

    def gaussianActivation(self):
        """Generate gaussian activation maps, based on sigmas and phosphene mapping"""
        Gauss = torch.exp(-0.5*((self.sigma**-2).view(self.batch_size,self.sigma.shape[1],1,1)*torch.unsqueeze(self.pMap,dim=0)**2))
        alpha = 1/(self.sigma*torch.sqrt(torch.tensor(math.pi, device=self.device)))
        gaussian = alpha.view(self.batch_size,alpha.shape[1],1,1)*Gauss
        normalization_factor = gaussian.amax(dim=(2,3),keepdim=True)
    
        gaussian = (gaussian / normalization_factor).nan_to_num(nan=0,posinf=0,neginf=0) #TODO: best way to do this?
        return gaussian

    def print_stats(self, stat_name, stat):
        print(f"""{stat_name}:
        size:   {stat.size()}
        min:    {stat.min():.2E}
        max:    {stat.max():.2E}
        mean:   {stat.mean():.2E}
        std:    {stat.std():.2E}""")

    def get_state(self):
        state = {
            'brightness': self.brightness.cpu().numpy(),
            'sigma': self.sigma.cpu().numpy(),
            'trace_fast': self.trace_fast.cpu().numpy(),
            'trace_slow': self.trace_slow.cpu().numpy()
        }
        return state

    def __call__(self,stim_amp,pw=None,freq=None):
        """Return phosphenes (2d image) based on current stimulation and previous state (self.activation, self.trace, self.sigma)"""

        self.batch_size = stim_amp.shape[0]

        # Update current state (self.activation, self.sigma) 
        # according to current stimulation and previous state 
        self.update(stim_amp,pw,freq)
        
        # Generate phosphene map
        phs = self.gaussianActivation()
        
        # Return phosphenes
        activated_phs = self.brightness.view(self.batch_size, self.brightness.shape[1],1,1)*phs
        output = torch.sum(activated_phs,dim=1)
        
        return output


if __name__ == '__main__':
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", type=str, default="webcam",
                    help="mode, webcam input or video")
    ap.add_argument("-vid", "--video_path", type=str, help="path to video to be processed")
    ap.add_argument("-save", "--save_path", type=str, help="path to save processed video to (webcam input is not saved")
    ap.add_argument("-nf", "--n_frames", type=int, help="number of frames to process")
    ap.add_argument("-nphos", "--n_phosphenes", type=int, help="number of phosphenes in simulator")
    ap.add_argument("-dev", "--device", type=str, default="cpu", help="device cpu or e.g. cuda:0")
    ap.add_argument("-t", "--time", action='store_true')
    ap.add_argument("-no-t", "--no_time", dest='time', action='store_true')

    ap.set_defaults(time=True)

    args = ap.parse_args()

    params = utils.load_params('config/params.yaml')
    # electrode_coords = utils.load_coords_from_yaml('config/grid_coords_valid.yaml', n_coords=100)
    # r, phi = init.init_from_cortex_full_view(params, electrode_coords)
    r, phi = init.init_probabilistically(params,n_phosphenes=args.n_phosphenes)
    
    simulator = GaussianSimulator(params, r, phi, batch_size=1, device=args.device)
    
    resolution = params['run']['resolution']
    
    if args.mode == "webcam":
        utils.webcam_demo(simulator, params, resolution=resolution)
    elif args.mode== "video":
        utils.process_video(simulator, params, args.video_path, args.save_path, args.n_frames, args.time, device=args.device)
