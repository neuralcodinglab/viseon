import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import numbers
import torchvision
from torchvision import transforms
import cv2 as cv
import utils



def convlayer(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    layer = [
        nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer  


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock, self).__init__()
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
        self.rgbConverter = utils.RGBConverter()
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
        self.model = nn.Sequential(*convlayer(in_channels,8,3,1,1),
                                   *convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   *convlayer(32,16,3,1,1),
                                   nn.Conv2d(16,out_channels,3,1,1),  
                                   nn.Tanh())
    def forward(self, x):
        self.out = self.model(x)
        x = self.out
        if self.binary_stimulation:
            x = x + torch.sign(x).detach() - x.detach() # (self-through estimator)
        stimulation = .5*(x+1)
        return stimulation    


class E2E_Encoder2(nn.Module):
    """
    Encoder that is used in Exp4, with a linear head. Each output indicates activation of a single electrode 
    """   
    def __init__(self, in_channels=3, out_channels=650,binary_stimulation=True):
        super(E2E_Encoder2, self).__init__()
             
        self.binary_stimulation = binary_stimulation
        
        # Model
        self.model = nn.Sequential(*convlayer(in_channels,8,3,1,1),
                                   *convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   *convlayer(32,16,3,1,0),
                                   *convlayer(16,1,3,1,0),
                                   nn.Flatten(),
                                   nn.Linear(28*28,out_channels),
                                   nn.Tanh())
    def forward(self, x):
        self.out = self.model(x)
        x = self.out
        if self.binary_stimulation:
            x = x + torch.sign(x).detach() - x.detach() # (self-through estimator)
        stimulation = .5*(x+1)
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
        self.model = nn.Sequential(*convlayer(in_channels,16,3,1,1),
                                   *convlayer(16,32,3,1,1),
                                   *convlayer(32,64,3,2,1),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   *convlayer(64,32,3,1,1),
                                   nn.Conv2d(32,out_channels,3,1,1), 
                                   self.out_activation)       

    def forward(self, x):
        return self.model(x)    
    
    
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

class Simulator2(object):
    """ Modular phosphene simulator that is used in experiment 4. Requires a predefined phosphene mapping. e.g. Tensor of 650 X 256 X 256 where 650 is the number of phosphenes and 256 X 256 is the resolution of the output image."""
    def __init__(self,pMap=None, pMap_from_file='training_configuration/model_parameters/phosphene_map_exp4.pt'):
        # Phospene mapping (should be of shape: n_phosphenes, res_x, res_y)
        if pMap is not None:
            self.pMap = pMap
        else:
            self.pMap = torch.load(pMap_from_file)
    
    def __call__(self,stim):
        return torch.einsum('ij, jkl -> ikl', stim, self.pMap).unsqueeze(dim=1) 

class Canny_Encoder(nn.Module):
    """
    Encoder class based on the OpenCV Canny edge detection implementation. 
    Outputs stimulation protocol with same size as input.
    
    low: low threshold
    high: high threshold
    sigma: size parameter for Gaussian blur
    """   
    def __init__(self, low=40, high=80, sigma=1.5, **kwargs):
        super(Canny_Encoder, self).__init__()
        
        # Parameters
        self.thres = nn.parameter.Parameter(torch.tensor([low,high]), requires_grad=False)
        self.sigma = nn.parameter.Parameter(torch.tensor(sigma), requires_grad=False)
        ksize      = np.round(4.*sigma+1.).astype(int) #rule of thumb
        dilation_kernel  = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        
        
        # Image operations
        self.Canny = lambda img: cv.Canny(img,low,high)
        self.blur  = lambda img: cv.GaussianBlur(img,(ksize,ksize),sigma)
        self.dilate= lambda img: cv.dilate(img,dilation_kernel)
        
    def forward(self, x):
        x = 255*(x-x.min())/(x.max()-x.min()) # standardize to 8-bit unsigned integer
        img = x.cpu().numpy().squeeze().astype('uint8')
        img = [self.blur(i)  for i in img]
        img = [self.Canny(i) for i in img]
        img = [torch.tensor(self.dilate(i)/255.,device=x.device).float() for i in img]
        self.out = torch.stack(img,dim=0).unsqueeze(dim=1)    
        return self.out
    
 # HED Model
class CropLayer(object):
    """Caffe layer for OpenCV HED model"""
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]   

class HED_Encoder(nn.Module):
    """
    Encoder class based on the OpenCV HED implementation. 
    Outputs stimulation protocol with same size as input.
    """   
    def __init__(self, **kwargs):
        super(HED_Encoder, self).__init__()
        
        # Parameters
        protoPath = "deploy.prototxt"
        modelPath = "hed_pretrained_bsds.caffemodel"
        self.net = cv.dnn.readNetFromCaffe(protoPath, modelPath)
        cv.dnn_registerLayer("Crop", CropLayer) # register our new layer with the model
    
    def forward(self, x):
        """ Note x is provided in [B,C,H,W] format where C should be RGB channels"""
        i = x.repeat(1,3,1,1) if x.shape[1]==1 else x[:,[2,1,0],:,:] # convert to BGR
        self.net.setInput(i.cpu().numpy())
        self.out = torch.tensor(self.net.forward(),device=x.device).float()
        return (self.out>0.5).float()
    


class Intensity_Encoder(nn.Module):
    """
    Encoder class that returns the thresholded pixel_intensity of the input (same size)
    """   
    def __init__(self, threshold=0.5, **kwargs):
        super(Intensity_Encoder, self).__init__()
        # Parameters
        self.thres = nn.parameter.Parameter(torch.tensor(threshold), requires_grad=False)
        
    def forward(self, x):
        self.out = x   
        return (self.out>self.thres).float()
    
    
    
    
  #### Old model   
# class E2E_CannyModel(nn.Module):
#     """Uses openCVs Canny edge detection module for image filtering. 
#     The edge map is converted to a stimulation map (by downsampling to n_phosphenes*n_phosphenes"""
#     def __init__(self,scale_factor,device,imsize=(128,128),ksize=(7,7),
#                  sigma=1,low=50,high=100,dilation=False):
#         super(E2E_CannyModel, self).__init__()
        
#         k_size           = (1,1) if not dilation else (3,3)
#         dilation_kernel  = cv.getStructuringElement(cv.MORPH_ELLIPSE,k_size)
        
#         self.device = device        
#         self.to_cv2_list = lambda image_tensor : [np.squeeze((255*img.cpu().numpy())).astype('uint8') for img in image_tensor]
#         self.gaus_blur   = lambda image_list : [cv.GaussianBlur(img,ksize=ksize,sigmaX=sigma) for img in image_list]
#         self.canny_edge  = lambda image_list : [cv.Canny(img,low,high) for img in image_list]
#         self.dilate      = lambda image_list : [cv.dilate(img,dilation_kernel) for img in image_list]
#         self.to_tensor   = lambda image_list : torch.tensor(image_list, device=device,dtype=torch.float32).unsqueeze(axis=1)
#         self.interpolate = lambda image_tensor : F.interpolate(image_tensor,scale_factor=scale_factor)
        
#         self.model = transforms.Compose([transforms.Lambda(self.to_cv2_list),
#                                              transforms.Lambda(self.gaus_blur),
#                                              transforms.Lambda(self.canny_edge),
#                                              transforms.Lambda(self.dilate),
#                                              transforms.Lambda(self.to_tensor),
#                                              transforms.Lambda(self.interpolate)])
                                                   
#     def forward(self, x):
#         return  self.model(x)/255  
    
