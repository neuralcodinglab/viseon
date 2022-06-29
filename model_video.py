from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import numbers
import torchvision
from torchvision import transforms
import cv2 as cv

from simulator.simulator import GaussianSimulator
from utils import RGBConverter

##TODO: remove, just for convenience and debugging
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

def convlayer3d(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    layer = [
        nn.Conv3d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer 

def deconvlayer3d(n_input, n_output, k_size=2, stride=2, padding=0, dilation=1, resample_out=None):
    layer = [
        nn.ConvTranspose3d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm3d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer

def convlayer2d(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    layer = [
        nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer  

class ResidualBlock3d(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm3d(n_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm3d(n_channels)
        self.resample_out = resample_out
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if self.resample_out:
            out = self.resample_out(out)
        return out

class ResidualBlock2d(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.resample_out = resample_out
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if self.resample_out:
            out = self.resample_out(out)
        return out

class VGG_Feature_Extractor(object):
    def __init__(self,layer_depth=4,device='cuda:0'):
        """Use the first <layer_depth> layers of the vgg16 model pretrained on imageNet as feature extractor.  
        When called, returns the feature maps of input image. (grayscale input is automatically converted to RGB"""
        model = torchvision.models.vgg16(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[0][:int(layer_depth)]).to(device)
        for child in [*self.feature_extractor]:
            for p in child.parameters():
                p.requires_grad = False
        self.rgbConverter = RGBConverter()
    def __call__(self,image):
        if image.shape[1]==1:
            image = self.rgbConverter(image)
            return self.feature_extractor(image)
        else:
            assert image.shape[1]==3
            return self.feature_extractor(image)    

class E2E_Encoder(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """   
    def __init__(self, in_channels=3, out_channels=1,binary_stimulation=True):
        super(E2E_Encoder, self).__init__()
             
        self.binary_stimulation = binary_stimulation
        
        # Model
        self.model = nn.Sequential(*convlayer2d(in_channels,8,3,1,1),
                                   *convlayer2d(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer2d(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   ResidualBlock2d(32, resample_out=None),
                                   ResidualBlock2d(32, resample_out=None),
                                   ResidualBlock2d(32, resample_out=None),
                                   ResidualBlock2d(32, resample_out=None),
                                   *convlayer2d(32,16,3,1,1),
                                   nn.Conv2d(16,out_channels,3,1,1),
                                   nn.Flatten(),
                                   nn.Linear(1024,1024),
                                   nn.ReLU()
                                #    nn.Softplus() #TODO: non-binary stimulation 
                                #    nn.Tanh()
                                #    nn.Sigmoid()
                                )

    def forward(self, x):
        self.out = self.model(x)
        # x = self.out
        if self.binary_stimulation:
            x = x + torch.sign(x).detach() - x.detach() # (self-through estimator) #TODO: non-binary stimulation 
            
        # stimulation = .5*(x+1) #why?
        # stimulation = stimulation*50.
        stimulation = self.out
        
        return stimulation    
    
class E2E_Decoder(nn.Module):
    """
    Simple non-generic phosphene decoder.
    in: (256x256) SVP representation
    out: (128x128) Reconstruction
    """   
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(E2E_Decoder, self).__init__()
             
        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]
        
        # Model
        self.model = nn.Sequential(*convlayer2d(in_channels,16,3,1,1),
                                   *convlayer2d(16,32,3,1,1),
                                   *convlayer2d(32,64,3,2,1),
                                   ResidualBlock2d(64),
                                   ResidualBlock2d(64),
                                   ResidualBlock2d(64),
                                   ResidualBlock2d(64),
                                   *convlayer2d(64,32,3,1,1),
                                   nn.Conv2d(32,out_channels,3,1,1), 
                                   self.out_activation)       

    def forward(self, x):
        return self.model(x)  

class ZhaoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ZhaoEncoder, self).__init__()

        self.model = nn.Sequential(
            *convlayer3d(in_channels,32,3,1,1, resample_out=nn.MaxPool3d(2,(0,2,2),dilation=(0,1,1))),
            *convlayer3d(32,48,3,1,1, resample_out=nn.MaxPool3d(2,(0,2,2),dilation=(0,1,1))),
            *convlayer3d(48,64,3,1,1),#, resample_out=nn.MaxPool3d(2,(0,2,2),dilation=(0,1,1))),
            *convlayer3d(64,64,3,1,1),
            # nn.Conv3d(64,out_channels,3,1,1),
            nn.Flatten(),
            nn.Linear(512,1024),
            nn.ReLU()
        )

    def forward(self, x):
        self.out = self.model(x)
        return self.out

class ZhaoDecoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=1):
        super(ZhaoDecoder, self).__init__()

        self.model = nn.Sequential(
            *deconvlayer3d(in_channels,48,2,(1,2,2),0,(0,1,1)),
            PrintLayer(),
            *deconvlayer3d(48,32,(1,2,2),0,(0,1,1)),
            PrintLayer(),
            *deconvlayer3d(32,32,2,(1,1,1),0,(0,0,0)),
            PrintLayer(),
            nn.Conv3d(32,out_channels,3,1,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        self.out = self.model(x)
        return self.out


class V2V_Encoder(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """   
    def __init__(self, in_channels=3, out_channels=1,binary_stimulation=True):
        super(V2V_Encoder, self).__init__()
             
        self.binary_stimulation = binary_stimulation
        
        # Model
        self.model = nn.Sequential(*convlayer3d(in_channels,8,3,1,1),
                                   *convlayer3d(8,16,3,1,1,resample_out=nn.MaxPool3d(2)),
                                   *convlayer3d(16,32,3,1,1,resample_out=nn.MaxPool3d(2)),
                                   ResidualBlock3d(32, resample_out=None),
                                   ResidualBlock3d(32, resample_out=None),
                                   ResidualBlock3d(32, resample_out=None),
                                   ResidualBlock3d(32, resample_out=None),
                                   *convlayer3d(32,16,3,1,1),
                                   nn.Conv3d(16,out_channels,3,1,1),
                                   nn.Flatten(),
                                   nn.Linear(1024,1024),
                                   nn.ReLU()
                                #    nn.Softplus() #TODO: non-binary stimulation 
                                #    nn.Tanh()
                                #    nn.Sigmoid()
                                )

    def forward(self, x):
        self.out = self.model(x)
        # x = self.out
        if self.binary_stimulation:
            x = x + torch.sign(x).detach() - x.detach() # (self-through estimator) #TODO: non-binary stimulation 
            
        stimulation = self.out
        
        return stimulation    
    
class V2V_Recon_Decoder(nn.Module):
    """
    Simple non-generic phosphene decoder.
    in: (256x256) SVP representation
    out: (128x128) Reconstruction
    """   
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(V2V_Recon_Decoder, self).__init__()
             
        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]
        
        # Model
        self.model = nn.Sequential(*convlayer2d(in_channels,16,3,1,1),
                                   *convlayer2d(16,32,3,1,1),
                                   *convlayer2d(32,64,3,2,1),
                                   ResidualBlock2d(64),
                                   ResidualBlock2d(64),
                                   ResidualBlock2d(64),
                                   ResidualBlock2d(64),
                                   *convlayer2d(64,32,3,1,1),
                                   nn.Conv2d(32,out_channels,3,1,1), 
                                   self.out_activation)       

    def forward(self, x):
        return self.model(x)    
    
class E2E_RealisticPhospheneSimulator(nn.Module):
    """A realistic simulator, using stimulation vectors to form a phosphene representation
    in: a 32x32 stimulation vector #TODO: flatten before inputting?
    out: 256x256 phosphene representation
    """
    def __init__(self, pMap,sigma_0, activation_mask, threshold, thresh_slope, args, device=torch.device('cuda:0')):
    #pMask,scale_factor=8, sigma=1.5,kernel_size=11, intensity=15, device=torch.device('cuda:0')):
        super(E2E_RealisticPhospheneSimulator, self).__init__()
        
        # use_cuda = False if device == 'cpu' else True

        self.simulator = GaussianSimulator(pMap, sigma_0, activation_mask, threshold, thresh_slope, **args)

    def forward(self, stimulation):
        # stim = stimulation.flatten(start_dim=1)
        # print(f"stim shape before entering simulator: {stimulation.shape}")
        phosphenes = self.simulator(stim=stimulation*100).clamp(0,150)
        # print(f"phosphene shape after flatten: {phosphenes.shape}")
        phosphenes = phosphenes/150  #phosphenes.max()
        
        phosphenes = phosphenes.view(phosphenes.shape[0], 1, phosphenes.shape[1], phosphenes.shape[2])
        return phosphenes


class E2E_PhospheneSimulator(nn.Module):
    """ Uses three steps to convert  the stimulation vectors to phosphene representation:
    1. Resizes the feature map (default: 32x32) to SVP template (256x256)
    2. Uses pMask to sample the phosphene locations from the SVP activation template
    2. Performs convolution with gaussian kernel for realistic phosphene simulations
    """
    def __init__(self,pMask,scale_factor=8, sigma=1.5,kernel_size=11, intensity=15, device=torch.device('cuda:0')):
        super(E2E_PhospheneSimulator, self).__init__()
        
        # Device
        self.device = device
        
        # Phosphene grid
        self.pMask = pMask
        self.up = nn.Upsample(mode="nearest",scale_factor=scale_factor)
        self.gaussian = self.get_gaussian_layer(kernel_size=kernel_size, sigma=sigma, channels=1)
        self.intensity = intensity 
    
    def get_gaussian_layer(self, kernel_size, sigma, channels):
        """non-trainable Gaussian filter layer for more realistic phosphene simulation"""

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter    

    def forward(self, stimulation):
        
        # Phosphene simulation
        phosphenes = self.up(stimulation)*self.pMask
        phosphenes = self.gaussian(F.pad(phosphenes, (5,5,5,5), mode='constant', value=0)) 
        return self.intensity*phosphenes    

class E2E_CannyModel(nn.Module):
    """Uses openCVs Canny edge detection module for image filtering. 
    The edge map is converted to a stimulation map (by downsampling to n_phosphenes*n_phosphenes"""
    def __init__(self,scale_factor,device,imsize=(128,128),ksize=(7,7),sigma=1,low=50,high=100):
        super(E2E_CannyModel, self).__init__()
        
        self.device = device        
        self.to_cv2_list = lambda image_tensor : [np.squeeze((255*img.cpu().numpy())).astype('uint8') for img in image_tensor]
        self.gaus_blur   = lambda image_list : [cv.GaussianBlur(img,ksize=ksize,sigmaX=sigma) for img in image_list]
        self.canny_edge  = lambda image_list : [cv.Canny(img,low,high) for img in image_list]
        self.to_tensor   = lambda image_list : torch.tensor(image_list, device=device,dtype=torch.float32).unsqueeze(axis=1)
        self.interpolate = lambda image_tensor : F.interpolate(image_tensor,scale_factor=scale_factor)
        
        self.model = transforms.Compose([transforms.Lambda(self.to_cv2_list),
                                             transforms.Lambda(self.gaus_blur),
                                             transforms.Lambda(self.canny_edge),
                                             transforms.Lambda(self.to_tensor),
                                             transforms.Lambda(self.interpolate)])
                                                   
    def forward(self, x):
        return  self.model(x)/255  
    
