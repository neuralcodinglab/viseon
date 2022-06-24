import torch
import random
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
import math
import cv2
import time

class GaussianSimulator(object):
    def __init__(self, pMap, sigma_0, sampling_mask, threshold=None, slope_thresh=None, batch_size=1, **kwargs):
        """ Initialize with:
        pMap (Pytorch tensor): a stack of phosphene mappings (i x j x k)
        sigma_0 (Pytorch tensor): a vector of phosphene sizes at start of simulation (i)
        sampling_mask (Pytorch tensor): a stack of (binary valued) receptive fields (i x j x k)
        For dimensions i = n_phosphenes, j = pixels_y , k = pixels_x, """
        
        # Phosphene mapping, sizes and receptive fields
        self.pMap = pMap
        self.device = pMap.device
        self.sigma_0 = sigma_0
        self.sampling_mask = sampling_mask
        assert (sampling_mask.device == self.device) and (sigma_0.device == self.device)
            
        self.batch_size = batch_size
        # Other parameters
        self.__dict__.update(kwargs)

        self.print_stats=False
        
        # if self.use_threshold: #TODO: check whether this can be here or needs to be in update b/o torch graph construction
        #     self.SIGMOID = lambda x: 1 / (1 + torch.exp(-x))
        #     self.SLOPE_FACTOR = lambda input, base, center: torch.pow(base,(input-center)/center)
        #     self.THRESHOLD_FACTOR = lambda input, base, center: torch.pow(base,-(input-center)/center)

        # Reset memory 
        self.reset()  
    
    def reset(self):
        """Reset Memory of previous timestep"""
        self.sigma = self.sigma_0.repeat(self.batch_size,1).to(self.device) 
        self.activation = torch.zeros(self.batch_size,len(self.pMap)).to(self.device)
        
    def update(self,stim): 
        """Adjust state as function of previous state and current stimulation. 
        NOTE: all parameters (such as intensity_decay) must be provided at initialization"""

        self.activation = stim #TODO: maybe multiply by constant? check!
        self.sigma = 10*torch.ones_like(stim)
        # # Bosking et al., 2017: AC = MD / (1 + np.exp(-slope*(I-I_50)))
        # AC = self.MD / torch.add(torch.exp(-self.slope_stim*(stim-self.I_half)),1)
        # # self.sigma = AC*self.sigma_0.repeat(self.batch_size,1) ### TODO: adjust temporal properties here, or not. Antonio thinks size is not affected by habituation, but rather seems to be due to using gaussians (lower intensity also leads to smaller perceived phosphene)
        # self.sigma = self.deg2pix_coeff*torch.einsum('ij, ij -> ij', AC, 1/self.sigma_0.repeat(self.batch_size,1))
        if self.print_stats:
            print("sigma:")
            print(f"min:    {self.sigma.min()}")
            print(f"max:    {self.sigma.max()}")
            print(f"mean:   {self.sigma.mean()}")
            print(f"std:    {self.sigma.std()}")
        
        if self.print_stats:
            print("activation:")
            print(f"min:    {self.activation.min()}")
            print(f"max:    {self.activation.max()}")
            print(f"mean:   {self.activation.mean()}")
            print(f"std:    {self.activation.std()}")
        
    def sample(self,img):
        """Create stimulation vector from activation mask (img) using the sampling mask"""
        img = img.repeat(self.batch_size,1,1)
        stim = (torch.einsum('ikl, jkl -> ij', img, self.sampling_mask)/self.sampling_mask.sum()*10).clamp(0.,1.)
        stim = 100*stim #normalize to range 0-100 micro-amps
        if self.print_stats:
            print("stimulation stats:")
            print(f"min:    {stim.min()}")
            print(f"max:    {stim.max()}")
            print(f"mean:   {stim.mean()}")
            print(f"std:    {stim.std()}")
        return stim

    def gaussianActivation(self):
        """Generate gaussian activation maps, based on sigmas and phosphene mapping"""
        Gauss = torch.exp(-0.5*torch.einsum('ij, jkl -> ijkl', self.sigma**-2, self.pMap**2))
        alpha = 1/(self.sigma*torch.sqrt(torch.tensor(math.pi, device=self.device)))
        gaussian = torch.einsum('ij, ijkl -> ijkl', alpha, Gauss)
        out = gaussian/gaussian.amax(dim=(2,3),keepdim=True) #normalize Gaussians
        return out

    def __call__(self,stim=None, img=None):
        """Return phosphenes (2d image) based on current stimulation and previous state (self.activation, self.trace, self.sigma)
        NOTE: either stimulation vector or input image (to sample from) must be provided!"""
        
        # Convert image to stimulation vector
        if stim is None:
            stim = self.sample(img)

        self.batch_size = stim.shape[0]

        # Update current state (self.activation, self.sigma) 
        # according to current stimulation and previous state 
        self.update(stim)
        
        # Generate phosphene map
        phs = self.gaussianActivation()
    
        # Return phosphenes
        return torch.einsum('ij, ijkl -> ikl', self.activation, phs)

# def get_deg2pix_coeff(display):
#     if not display:
#         print("assuming some standard values for display")
#         display = {
#             'screen_resolution'   : [1920,1080],
#             'screen_size'         : [294,166],
#             'dist_to_screen'      : 600
#         } 

#     screen_diagonal = np.sqrt(display['screen_size'][0]**2+display['screen_size'][1]**2)
#     screen_ppmm = np.sqrt(display['screen_resolution'][0]**2 + display['screen_resolution'][1]**2)/screen_diagonal
#     pixel_size = 1/screen_ppmm
#     deg_per_pixel = np.degrees(np.arctan(pixel_size/display['dist_to_screen']))

#     return 1/deg_per_pixel


def init_a_bit_less(args=None, n_phosphenes=32*32, resolution=(256,256), display=None, use_cuda=False):
    ### 0. User settings
    
    # General settings
    N_PHOSPHENES = n_phosphenes
    RESOLUTION = resolution
    USE_CUDA= use_cuda

    # Scaling factors and coefficients for realistic phosphene appearance
    # DEG2PIX_COEFF = get_deg2pix_coeff(display)
    # print(f"1 dva shown as {DEG2PIX_COEFF} pixels")
    
    # # Spatial phosphene characteristics
    SAMPLE_RADIAL_DISTR  = lambda : np.random.rand()**1 ##### TODO: PLAUSIBLE SPATIAL DISTRIBUTION OF PHOSPHENES
    ECCENTRICITY_SCALING = lambda r: 10#17.3/(0.75+r) #degrees per mm activated cortex, Horten & Hoyt
    # # DEG2PIX_COEFF*(SCALING_FACTOR*(EFF_ACTIVATION/(17.3/(0.75+r)))) #PLAUSIBLE CORTICAL MAGNIFICATION: Horten & Holt's formula
    # GABOR_ORIENTATION = lambda : np.random.rand()*2*math.pi #Rotation of ellips

    # Receptive field parameters (to generate activation mask)
    RECEPTIVE_FIELD_SIZE = 4
    USE_RELATIVE_RF_SIZE = True

    # if not args:
    #     args = {
    #     # processing images over time?
    #     'video'             : False,
    #     # habituation params
    #     'intensity_decay'   : 0.4,
    #     'input_effect'      : 1,
    #     'trace_decay'       : 0.99997,
    #     'trace_increase'    : 0.004,
    #     # current strength effect on size (Bosking et al., 2017)
    #     'MD'                : 0.7,  # predicted maximum diameter of activated cortex in mm
    #     'slope_stim'        : 0.05,  # slope in mm/mu-A
    #     'I_half'            : 20,  # mu-A
    #     # Gabor filters
    #     'gabor_filtering'   :False,
    #     'gamma'             :1.5,  
    #     # Stimulation threshold
    #     'use_threshold'     :False,
    #     'range_slope_thresh':(0.5,4),  # slope of psychometric curve
    #     'range_thresh'      :(1,100),  # range of threshold values
    #     # Pulse width (pw), frequency (freq) & train duration (td) input
    #     # 'calibrated' values, deviation influences slope & theshold of psychometric curve 
    #     'pw_default'        : 200,
    #     'freq_default'      : 200,
    #     'td_default'        : 125,
    #     'pw_change'         : 1.5,
    #     'freq_change'       : 1.5,
    #     'td_change'         : 2.
    # }

    # args['print_stats'] = False #TODO: remove, for debugging purposes
    # args['deg2pix_coeff'] = DEG2PIX_COEFF
    ###    
    ## 1. Generate phosphene mapping, sigmas and activation mask
    
    # Device
    device = 'cuda:0' if USE_CUDA else 'cpu'

    # Cartesian coordinate system for the visual field
    x = torch.arange(RESOLUTION[0]).float()
    y = torch.arange(RESOLUTION[1]).float()
    grid = torch.meshgrid(y, x, indexing='ij')

    # Phosphene mapping (radial distances to center of phosphene) 
    pMap = torch.zeros(N_PHOSPHENES, *RESOLUTION, device=device)

    # Sigma at start of simulation
    sigma_0 = torch.zeros(N_PHOSPHENES,device=device)

    for i in range(N_PHOSPHENES): ##TODO: I think this can be done without a for loop as a vector operation

        # Polar coordinates
        phi = np.pi *2 * np.random.rand(1)
        r = SAMPLE_RADIAL_DISTR() * (RESOLUTION[0] //2)
        # Convert to cartesian indices
        x_offset = np.round(np.cos(phi)*r) + RESOLUTION[1]//2
        y_offset = np.round(np.sin(phi)*r) + RESOLUTION[0]//2
        x_offset = np.clip(int(x_offset),0,RESOLUTION[1]-1)
        y_offset = np.clip(int(y_offset),0,RESOLUTION[0]-1)
        
        # Calculate distance map for every element wrt center of phosphene
        pMap[i] = torch.sqrt((grid[0]-y_offset)**2 + (grid[1]-x_offset)**2)

        sigma_0[i] = torch.tensor(ECCENTRICITY_SCALING(r/RESOLUTION[0])) 

    # Generate activation mask
    if USE_RELATIVE_RF_SIZE:
        # activation_mask = (pMap.permute(1,2,0) < sigma_0 * RECEPTIVE_FIELD_SIZE).permute(2,0,1).float()
        activation_mask = (pMap.permute(1,2,0) < 68*(1*(1/sigma_0)) * RECEPTIVE_FIELD_SIZE).permute(2,0,1).float()
    else: 
        activation_mask = (pMap < RECEPTIVE_FIELD_SIZE).float()

    print(f'device: {device}')
    # for key in args:
    #     args[key] = torch.tensor([args[key]], dtype=torch.float, device=device)

    threshold=None
    thresh_slope=None
    args={'dummy':0}
    return pMap,sigma_0, activation_mask, threshold, thresh_slope, args

def init_simulator(args=None, n_phosphenes=32*32, resolution=(256,256), display=None, use_cuda=False):
    ### 0. User settings
    
    # General settings
    N_PHOSPHENES = n_phosphenes
    RESOLUTION = resolution
    USE_CUDA= use_cuda

    # Scaling factors and coefficients for realistic phosphene appearance
    # EFF_ACTIVATION = 1 #diameter of the area in which neurons are activated - in theory this is in mm
    DEG2PIX_COEFF = 68#get_deg2pix_coeff(display)
    print(f"1 dva shown as {DEG2PIX_COEFF} pixels")
    # SCALING_FACTOR = 0.31 #used to get realistic phosphene sizes, based on Vurro et al. 2014, although that's a thalamic prosthesis

    # Spatial phosphene characteristics
    SAMPLE_RADIAL_DISTR  = lambda : np.random.rand()**2 ##### TODO: PLAUSIBLE SPATIAL DISTRIBUTION OF PHOSPHENES
    ECCENTRICITY_SCALING = lambda r: 17.3/(0.75+r) #degrees per mm activated cortex, Horten & Hoyt
    #DEG2PIX_COEFF*(SCALING_FACTOR*(EFF_ACTIVATION/(17.3/(0.75+r)))) #PLAUSIBLE CORTICAL MAGNIFICATION: Horten & Holt's formula
    GABOR_ORIENTATION = lambda : np.random.rand()*2*math.pi #Rotation of ellips

    # Receptive field parameters (to generate activation mask)
    RECEPTIVE_FIELD_SIZE = 4
    USE_RELATIVE_RF_SIZE = True

    if not args:
        args = {
        # processing images over time?
        'video'             : False, #TODO: if true, only take full badges as input
        # habituation params
        'intensity_decay'   : 0.4,
        'input_effect'      : 1.,
        'trace_decay'       : 0.99997,
        'trace_increase'    : 0.004,
        # current strength effect on size (Bosking et al., 2017)
        'MD'                : 0.7,  # TODO: originally set to 2?, predicted maximum diameter of activated cortex in mm
        'slope_stim'        : 0.05,  # slope in mm/mA
        'I_half'            : 20,  # mA
        # Gabor filters
        'gabor_filtering'   : True,
        'gamma'             : 1.5,  
        # Stimulation threshold
        'use_threshold'     : True,
        'range_slope_thresh': (0.5,4),  # slope of psychometric curve
        'range_thresh'      : (1,100),  # range of threshold values
        # Pulse width (pw), frequency (freq) & train duration (td) input
        # 'calibrated' values, deviation influences slope & theshold of psychometric curve 
        'pw_default'        : 200,
        'freq_default'      : 200,
        'td_default'        : 125,
        'pw_change'         : 1.5,
        'freq_change'       : 1.5,
        'td_change'         : 2.
    }

    args['print_stats'] = True #TODO: remove, for debugging purposes
    args['deg2pix_coeff'] = DEG2PIX_COEFF

    ###    
    ## 1. Generate phosphene mapping, sigmas and activation mask
    
    # Device
    device = 'cuda:0' if USE_CUDA else 'cpu'

    # Cartesian coordinate system for the visual field
    x = torch.arange(RESOLUTION[0]).float()
    y = torch.arange(RESOLUTION[1]).float()
    grid = torch.meshgrid(y, x, indexing='ij')

    # Phosphene mapping (radial distances to center of phosphene) 
    pMap = torch.zeros(N_PHOSPHENES, *RESOLUTION, device=device)

    # Sigma at start of simulation
    sigma_0 = torch.zeros(N_PHOSPHENES,device=device)

    # Stimulation threshold for each electrode to elicit a phosphene
    # threshold = torch.zeros(N_PHOSPHENES,device=device)
    # thresh_slope = torch.zeros(N_PHOSPHENES,device=device)

    if args['use_threshold']:
        threshold = torch.zeros(N_PHOSPHENES,device=device)
        thresh_slope = torch.zeros(N_PHOSPHENES,device=device)
    else:
        threshold = None
        thresh_slope = None

    for i in range(N_PHOSPHENES): 

        # Polar coordinates
        phi = np.pi *2 * np.random.rand(1)
        r = SAMPLE_RADIAL_DISTR() * (RESOLUTION[0] //2)
        # Convert to cartesian indices
        x_offset = np.round(np.cos(phi)*r) + RESOLUTION[1]//2
        y_offset = np.round(np.sin(phi)*r) + RESOLUTION[0]//2
        x_offset = np.clip(int(x_offset),0,RESOLUTION[1]-1)
        y_offset = np.clip(int(y_offset),0,RESOLUTION[0]-1)
        
        # Calculate distance map for every element wrt center of phosphene
        if args['gabor_filtering']:
            theta = GABOR_ORIENTATION()
            y = grid[0]-y_offset
            x = grid[1]-x_offset
            y_rotated = (-x * np.sin(theta)) + (y * np.cos(theta))
            x_rotated = (x * np.cos(theta)) + (y * np.sin(theta))
            pMap[i] = torch.sqrt(x_rotated**2 + (args['gamma']**2)*y_rotated**2)
        else:
            pMap[i] = torch.sqrt((grid[0]-y_offset)**2 + (grid[1]-x_offset)**2)

        if args['use_threshold']:
            threshold[i] = torch.tensor(args['range_thresh'][0]+(np.random.power(0.4)*(args['range_thresh'][1]-args['range_thresh'][0])))  # threshold dist based on Schmidt et al., 1996 
            thresh_slope[i] = torch.tensor(np.random.uniform(low=args['range_slope_thresh'][0], high=args['range_slope_thresh'][1]))  # threshold dist based on Schmidt et al., 1996 

        sigma_0[i] = torch.tensor(ECCENTRICITY_SCALING((1/DEG2PIX_COEFF)*r)) 

    # Generate activation mask
    if USE_RELATIVE_RF_SIZE:
        # activation_mask = (pMap.permute(1,2,0) < sigma_0 * RECEPTIVE_FIELD_SIZE).permute(2,0,1).float()
        activation_mask = (pMap.permute(1,2,0) < DEG2PIX_COEFF*(1*(1/sigma_0)) * RECEPTIVE_FIELD_SIZE).permute(2,0,1).float()
    else: 
        activation_mask = (pMap < RECEPTIVE_FIELD_SIZE).float()

        
    ## 2. Initialize simulator
    simulator = GaussianSimulator(pMap,sigma_0, activation_mask, threshold, thresh_slope, **args)
    #intensity_decay originally 0.4

    return simulator


def webcam_demo(simulator, resolution=(256,256), use_cuda=False):
    assert simulator.batch_size == 1, "for webcam demo use batch size=1"
    # video_capture 
    IN_VIDEO = 0 # use 0 for webcam, or string with video path"
    FRAMERATE = 35
    RESOLUTION = resolution

    # Canny threshold
    T_HIGH = 120

    # device
    device = 'cuda:0' if use_cuda else 'cpu'

    prev = 0
    cap = cv2.VideoCapture(IN_VIDEO)
    ret, frame = cap.read()

    while(ret):

        # Capture the video frame by frame
        ret, frame = cap.read()

        time_elapsed = time.time() - prev
        # ret, image = cap.read()
        if time_elapsed > 1./FRAMERATE:
            prev = time.time()

            # Create Canny edge detection mask
            frame = cv2.resize(frame, RESOLUTION)
            canny = cv2.Canny(frame,T_HIGH//2,T_HIGH)
            canny = torch.tensor(canny, device=device).float()

            # Convert frame to grayscale
            frame = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),device=device).float()
        
            # Generate phosphenes 
            phs = simulator(img=canny).clamp(0,150)
            phs = torch.squeeze(phs)
            phs = 255*phs/150

            # Concatenate results
            cat = torch.cat([frame, canny, phs], dim=1)
            cat = cat.cpu().numpy().astype('uint8')

            # Display the resulting frame
            cv2.imshow('Simulator', cat)

        # the 'q' button is set as the quit button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    
    simulator = init_simulator(n_phosphenes=16*16, resolution=(512,512))

    webcam_demo(simulator, resolution=(512,512))