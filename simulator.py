import torch
import random
from matplotlib import pyplot as plt
import numpy as np
import math
import cv2
import time

class GaussianSimulator(object):
    def __init__(self, pMap, sigma_0, sampling_mask, threshold=None, slope_thresh=None, **kwargs):
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
        self.threshold = threshold
        self.slope_thresh = slope_thresh
        assert (sampling_mask.device == self.device) and (sigma_0.device == self.device) and (threshold.device == self.device) and (slope_thresh.device == self.device) 
            
        # Other parameters
        self.__dict__.update(kwargs)

        # if self.use_threshold:
        #     self.SIGMOID = lambda x: 1 / (1 + torch.exp(-x))
        #     self.SLOPE_FACTOR = lambda input, base, center: torch.pow(base,(input-center)/center)
        #     self.THRESHOLD_FACTOR = lambda input, base, center: torch.pow(base,-(input-center)/center)

        # Reset memory 
        self.reset()  
    
    def reset(self):
        """Reset Memory of previous timestep"""
        self.sigma = self.sigma_0
        self.activation = torch.zeros(len(self.pMap)).to(self.device)
        self.trace = torch.zeros(len(self.pMap)).to(self.device)
        
    def update(self,stim): 
        """Adjust state as function of previous state and current stimulation. 
        NOTE: all parameters (such as intensity_decay) must be provided at initialization"""

        # update activation using trace for habituation 
        print(stim.shape)
        self.activation = self.intensity_decay*self.activation + self.input_effect*(stim-self.trace)  
        self.activation = torch.max(self.activation,torch.zeros_like(self.activation))  # don't allow negative activations
        self.trace = self.trace_decay*self.trace + self.trace_increase*stim  # update trace

        # Bosking et al., 2017: AC = MD / (1 + np.exp(-slope*(I-I_50)))
        AC = self.MD / (1+torch.exp(-self.slope_stim*(stim-self.I_half)))
        self.sigma = AC*self.sigma_0 ### TODO: adjust temporal properties here, or not. Antonio thinks size is not affected by habituation, but rather seems to be due to using gaussians (lower intensity also leads to smaller perceived phosphene)

        if self.use_threshold:
            self.SIGMOID = lambda x: 1 / (1 + torch.exp(-x))
            self.SLOPE_FACTOR = lambda input, base, center: torch.pow(base,(input-center)/center)
            self.THRESHOLD_FACTOR = lambda input, base, center: torch.pow(base,-(input-center)/center)

            # Use psychometric curve to determine whether phosphenes are seen
            slope_diff, threshold_diff = self.psychometricCurve(torch.ones_like(stim)*self.pw_default, torch.ones_like(stim)*self.freq_default, torch.ones_like(stim)*self.td_default) # TODO: these three parameters as input to this function, so it can be dynamic. An idea might be making a stimulation class...
            slope = self.slope_thresh*slope_diff
            threshold = self.threshold*threshold_diff # TODO: better names cause this is getting confusing
            probs = self.SIGMOID(slope*(self.activation-threshold))
            spikes = torch.rand(probs.shape[0]) < probs
            self.activation = spikes*self.activation  # TODO: effect of threshold over time: multiply here with activation, or with copy of activation (not sure how this works combined with the trace)
        
    def sample(self,img):
        """Create stimulation vector from activation mask (img) using the sampling mask"""
        stim = torch.einsum('jk, ijk -> i', img, self.sampling_mask)
        return stim

    def gaussianActivation(self):
        """Generate gaussian activation maps, based on sigmas and phosphene mapping"""
        Gauss = torch.exp(-0.5*torch.einsum('i, ijk -> ijk', self.sigma**-2, self.pMap**2))
        alpha = 1/(self.sigma*torch.sqrt(torch.tensor(math.pi, device=self.device)))
        return torch.einsum('i, ijk -> ijk', alpha, Gauss)

    def psychometricCurve(self, pulse_width, frequency, train_duration):
        pw_slope = self.SLOPE_FACTOR(pulse_width, self.pw_change,self.pw_default)
        pw_threshold = self.THRESHOLD_FACTOR(pulse_width,self.pw_change,self.pw_default)

        freq_slope = self.SLOPE_FACTOR(frequency,self.freq_change,self.freq_default)
        freq_threshold = self.THRESHOLD_FACTOR(frequency,self.freq_change,self.freq_default)

        td_slope = self.SLOPE_FACTOR(train_duration,self.td_change,self.td_default)
        td_threshold = self.THRESHOLD_FACTOR(train_duration,self.td_change,self.td_default)

        return pw_slope*freq_slope*td_slope, pw_threshold*freq_threshold*td_threshold

    def __call__(self,stim=None, img=None):
        """Return phosphenes (2d image) based on current stimulation and previous state (self.activation, self.trace, self.sigma)
        NOTE: either stimulation vector or input image (to sample from) must be provided!"""
        
        # Convert image to stimulation vector
        if stim is None:
            stim = self.sample(img)
        
        # Update current state (self.activation, self.sigma) 
        # according to current stimulation and previous state 
        self.update(stim)
        
        # Generate phosphene map
        phs = self.gaussianActivation()
        
        # Return phosphenes
        return torch.einsum('i, ijk -> jk', self.activation, phs)

def init_a_bit_less(args=None, n_phosphenes=32*32, resolution=(256,256), use_cuda=False):
    ### 0. User settings
    
    # General settings
    N_PHOSPHENES = n_phosphenes
    RESOLUTION = resolution
    USE_CUDA= use_cuda

    # Scaling factors and coefficients for realistic phosphene appearance
    EFF_ACTIVATION = 1 #diameter of the area in which neurons are activated - in theory this is in mm
    DEG2PIX_COEFF = 50/4 #### TODO: set parameters, use code from deg2pix notebook

    # Spatial phosphene characteristics
    SAMPLE_RADIAL_DISTR  = lambda : np.random.rand()**4 ##### TODO: PLAUSIBLE SPATIAL DISTRIBUTION OF PHOSPHENES
    # ECCENTRICITY_SCALING = lambda r:  2 * r + 0.5    #### ARTIFICIAL CORTICAL MAGNIFICATION
    ECCENTRICITY_SCALING = lambda r: DEG2PIX_COEFF*(EFF_ACTIVATION/(17.3/(0.75+r))) #PLAUSIBLE CORTICAL MAGNIFICATION: Horten & Holt's formula
    GABOR_ORIENTATION = lambda : np.random.rand()*2*math.pi #Rotation of ellips

    # Receptive field parameters (to generate activation mask)
    RECEPTIVE_FIELD_SIZE = 4
    USE_RELATIVE_RF_SIZE = True

    if not args:
        args = {
        # habituation params
        'intensity_decay'   : 0.05,
        'input_effect'      : 0.75,
        'trace_decay'       : 0.9,
        'trace_increase'    : 0.1,
        # current strength effect on size (Bosking et al., 2017)
        'MD'                : 2,  # predicted maximum diameter of activated cortex in mm
        'slope_stim'        : 0.55,  # slope in mm/mA
        'I_half'            : 5,  # mA
        # Gabor filters
        'gabor_filtering'   :True,
        'gamma'             :1.5,  
        # Stimulation threshold
        'use_threshold'     :True,
        'range_slope_thresh':(0.5,4),  # slope of psychometric curve
        'range_thresh'      :(1,100),  # range of threshold values
        # Pulse width (pw), frequency (freq) & train duration (td) input
        # 'calibrated' values, deviation influences slope & theshold of psychometric curve 
        'pw_default'        : 200,
        'freq_default'      : 200,
        'td_default'        : 125,
        'pw_change'         : 1.5,
        'freq_change'       : 1.5,
        'td_change'         : 2.
    }

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
    threshold = torch.zeros(N_PHOSPHENES,device=device)
    thresh_slope = torch.zeros(N_PHOSPHENES,device=device)

    for i in range(N_PHOSPHENES): ##TODO: I think this can be done without a for loop as a vector operation

        # Polar coordinates
        phi = np.pi *2 * np.random.rand(1)
        r = SAMPLE_RADIAL_DISTR() * RESOLUTION[0] 
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

        sigma_0[i] = torch.tensor(ECCENTRICITY_SCALING(r/RESOLUTION[0])) 

    # Generate activation mask
    if USE_RELATIVE_RF_SIZE:
        activation_mask = (pMap.permute(1,2,0) < sigma_0 * RECEPTIVE_FIELD_SIZE).permute(2,0,1).float()
    else: 
        activation_mask = (pMap < RECEPTIVE_FIELD_SIZE).float()

    return pMap,sigma_0, activation_mask, threshold, thresh_slope, args

def init_simulator(args=None, n_phosphenes=32*32, resolution=(256,256), use_cuda=False):
    ### 0. User settings
    
    # General settings
    N_PHOSPHENES = n_phosphenes
    RESOLUTION = resolution
    USE_CUDA= use_cuda

    # Scaling factors and coefficients for realistic phosphene appearance
    EFF_ACTIVATION = 1 #diameter of the area in which neurons are activated - in theory this is in mm
    DEG2PIX_COEFF = 50/4 #### TODO: set parameters, use code from deg2pix notebook

    # Spatial phosphene characteristics
    SAMPLE_RADIAL_DISTR  = lambda : np.random.rand()**4 ##### TODO: PLAUSIBLE SPATIAL DISTRIBUTION OF PHOSPHENES
    # ECCENTRICITY_SCALING = lambda r:  2 * r + 0.5    #### ARTIFICIAL CORTICAL MAGNIFICATION
    ECCENTRICITY_SCALING = lambda r: DEG2PIX_COEFF*(EFF_ACTIVATION/(17.3/(0.75+r))) #PLAUSIBLE CORTICAL MAGNIFICATION: Horten & Holt's formula
    GABOR_ORIENTATION = lambda : np.random.rand()*2*math.pi #Rotation of ellips

    # Receptive field parameters (to generate activation mask)
    RECEPTIVE_FIELD_SIZE = 4
    USE_RELATIVE_RF_SIZE = True

    if not args:
        args = {
        # habituation params
        'intensity_decay'   : 0.05,
        'input_effect'      : 0.75,
        'trace_decay'       : 0.9,
        'trace_increase'    : 0.1,
        # current strength effect on size (Bosking et al., 2017)
        'MD'                : 2,  # predicted maximum diameter of activated cortex in mm
        'slope_stim'        : 0.55,  # slope in mm/mA
        'I_half'            : 5,  # mA
        # Gabor filters
        'gabor_filtering'   :True,
        'gamma'             :1.5,  
        # Stimulation threshold
        'use_threshold'     :True,
        'range_slope_thresh':(0.5,4),  # slope of psychometric curve
        'range_thresh'      :(1,100),  # range of threshold values
        # Pulse width (pw), frequency (freq) & train duration (td) input
        # 'calibrated' values, deviation influences slope & theshold of psychometric curve 
        'pw_default'        : 200,
        'freq_default'      : 200,
        'td_default'        : 125,
        'pw_change'         : 1.5,
        'freq_change'       : 1.5,
        'td_change'         : 2.
    }

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
    threshold = torch.zeros(N_PHOSPHENES,device=device)
    thresh_slope = torch.zeros(N_PHOSPHENES,device=device)

    for i in range(N_PHOSPHENES): ##TODO: I think this can be done without a for loop as a vector operation

        # Polar coordinates
        phi = np.pi *2 * np.random.rand(1)
        r = SAMPLE_RADIAL_DISTR() * RESOLUTION[0] 
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

        sigma_0[i] = torch.tensor(ECCENTRICITY_SCALING(r/RESOLUTION[0])) 

    # Generate activation mask
    if USE_RELATIVE_RF_SIZE:
        activation_mask = (pMap.permute(1,2,0) < sigma_0 * RECEPTIVE_FIELD_SIZE).permute(2,0,1).float()
    else: 
        activation_mask = (pMap < RECEPTIVE_FIELD_SIZE).float()

        
    ## 2. Initialize simulator
    simulator = GaussianSimulator(pMap,sigma_0, activation_mask, threshold, thresh_slope, **args) #TODO: parameter tuning
    #intensity_decay originally 0.4

    return simulator


def webcam_demo(simulator, resolution=(256,256), use_cuda=False):
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
            phs = simulator(img=canny).clamp(0,500)
            phs = 255*phs/phs.max()

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
    
    simulator = init_simulator()

    webcam_demo(simulator)
    # ### 0. User settings
    
    # # General settings
    # N_PHOSPHENES = 32*32
    # RESOLUTION = (256,256)
    # USE_CUDA= False

    # # Scaling factors and coefficients for realistic phosphene appearance
    # EFF_ACTIVATION = 1 #diameter of the area in which neurons are activated - in theory this is in mm
    # DEG2PIX_COEFF = 50/4 #### TODO: set parameters, use code from deg2pix notebook

    # # Spatial phosphene characteristics
    # SAMPLE_RADIAL_DISTR  = lambda : np.random.rand()**4 ##### TODO: PLAUSIBLE SPATIAL DISTRIBUTION OF PHOSPHENES
    # # ECCENTRICITY_SCALING = lambda r:  2 * r + 0.5    #### ARTIFICIAL CORTICAL MAGNIFICATION
    # ECCENTRICITY_SCALING = lambda r: DEG2PIX_COEFF*(EFF_ACTIVATION/(17.3/(0.75+r))) #PLAUSIBLE CORTICAL MAGNIFICATION: Horten & Holt's formula
    # GABOR_ORIENTATION = lambda : np.random.rand()*2*math.pi #Rotation of ellips

    # # Receptive field parameters (to generate activation mask)
    # RECEPTIVE_FIELD_SIZE = 4
    # USE_RELATIVE_RF_SIZE = True
    
    # # video_capture 
    # IN_VIDEO = 0 # use 0 for webcam, or string with video path"
    # FRAMERATE = 35
    
    # # Canny threshold
    # T_HIGH = 120
    
    # # other params to pass to simulator
    # args = {
    #     # habituation params
    #     'intensity_decay'   : 0.05,
    #     'input_effect'      : 0.75,
    #     'trace_decay'       : 0.9,
    #     'trace_increase'    : 0.1,
    #     # current strength effect on size (Bosking et al., 2017)
    #     'MD'                : 2,  # predicted maximum diameter of activated cortex in mm
    #     'slope_stim'        : 0.55,  # slope in mm/mA
    #     'I_half'            : 5,  # mA
    #     # Gabor filters
    #     'gabor_filtering'   :True,
    #     'gamma'             :1.5,  
    #     # Stimulation threshold
    #     'use_threshold'     :True,
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

    # ###    
    # ## 1. Generate phosphene mapping, sigmas and activation mask
    
    # # Device
    # device = 'cuda:0' if USE_CUDA else 'cpu'

    # # Cartesian coordinate system for the visual field
    # x = torch.arange(RESOLUTION[0]).float()
    # y = torch.arange(RESOLUTION[1]).float()
    # grid = torch.meshgrid(y, x)

    # # Phosphene mapping (radial distances to center of phosphene) 
    # pMap = torch.zeros(N_PHOSPHENES, *RESOLUTION, device=device)

    # # Sigma at start of simulation
    # sigma_0 = torch.zeros(N_PHOSPHENES,device=device)

    # # Stimulation threshold for each electrode to elicit a phosphene
    # threshold = torch.zeros(N_PHOSPHENES,device=device)
    # thresh_slope = torch.zeros(N_PHOSPHENES,device=device)

    # for i in range(N_PHOSPHENES): ##TODO: I think this can be done without a for loop as a vector operation

    #     # Polar coordinates
    #     phi = np.pi *2 * np.random.rand(1)
    #     r = SAMPLE_RADIAL_DISTR() * RESOLUTION[0] 
    #     # Convert to cartesian indices
    #     x_offset = np.round(np.cos(phi)*r) + RESOLUTION[1]//2
    #     y_offset = np.round(np.sin(phi)*r) + RESOLUTION[0]//2
    #     x_offset = np.clip(int(x_offset),0,RESOLUTION[1]-1)
    #     y_offset = np.clip(int(y_offset),0,RESOLUTION[0]-1)
        
    #     # Calculate distance map for every element wrt center of phosphene
    #     if args['gabor_filtering']:
    #         theta = GABOR_ORIENTATION()
    #         y = grid[0]-y_offset
    #         x = grid[1]-x_offset
    #         y_rotated = (-x * np.sin(theta)) + (y * np.cos(theta))
    #         x_rotated = (x * np.cos(theta)) + (y * np.sin(theta))
    #         pMap[i] = torch.sqrt(x_rotated**2 + (args['gamma']**2)*y_rotated**2)
    #     else:
    #         pMap[i] = torch.sqrt((grid[0]-y_offset)**2 + (grid[1]-x_offset)**2)

    #     if args['use_threshold']:
    #         threshold[i] = args['range_thresh'][0]+(np.random.power(0.4)*(args['range_thresh'][1]-args['range_thresh'][0]))  # threshold dist based on Schmidt et al., 1996 
    #         thresh_slope[i] = np.random.uniform(low=args['range_slope_thresh'][0], high=args['range_slope_thresh'][1])  # threshold dist based on Schmidt et al., 1996 

    #     sigma_0[i] = torch.tensor(ECCENTRICITY_SCALING(r/RESOLUTION[0])) 

    # #TODO: take out this test
    # # plt.hist(threshold.detach().numpy(),bins=20)
    # # plt.show()

    # # Generate activation mask
    # if USE_RELATIVE_RF_SIZE:
    #     activation_mask = (pMap.permute(1,2,0) < sigma_0 * RECEPTIVE_FIELD_SIZE).permute(2,0,1).float()
    # else: 
    #     activation_mask = (pMap < RECEPTIVE_FIELD_SIZE).float()

        
    # ## 2. Initialize simulator
    # simulator = GaussianSimulator(pMap,sigma_0, activation_mask, threshold, thresh_slope, **args) #TODO: parameter tuning
    # #intensity_decay originally 0.4

    # ## 3. run webcam demo
    # prev = 0
    # cap = cv2.VideoCapture(IN_VIDEO)
    # ret, frame = cap.read()

    # while(ret):

    #     # Capture the video frame by frame
    #     ret, frame = cap.read()

    #     time_elapsed = time.time() - prev
    #     # ret, image = cap.read()
    #     if time_elapsed > 1./FRAMERATE:
    #         prev = time.time()

    #         # Create Canny edge detection mask
    #         frame = cv2.resize(frame, RESOLUTION)
    #         canny = cv2.Canny(frame,T_HIGH//2,T_HIGH)
    #         canny = torch.tensor(canny, device=device).float()

    #         # Convert frame to grayscale
    #         frame = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),device=device).float()
        
    #         # Generate phosphenes 
    #         phs = simulator(img=canny).clamp(0,500)
    #         phs = 255*phs/phs.max()

    #         # Concatenate results
    #         cat = torch.cat([frame, canny, phs], dim=1)
    #         cat = cat.cpu().numpy().astype('uint8')

    #         # Display the resulting frame
    #         cv2.imshow('Simulator', cat)

    #     # the 'q' button is set as the quit button
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # # Destroy all the windows
    # cv2.destroyAllWindows()