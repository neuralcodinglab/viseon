import cv2
import torch
from matplotlib.pyplot import axes
import numpy as np

def canny_processor(frame, threshold_low, threshold_high):
    canny = cv2.Canny(frame,threshold_low,threshold_high)
    return canny

def sobel_processor(frame):
    grad_x = cv2.Sobel(frame, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(frame, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    xy = np.stack([grad_x,grad_y])
    grad = np.linalg.norm(xy,axis=0)
    # abs_grad_x = cv2.convertScaleAbs(grad_x)
    # abs_grad_y = cv2.convertScaleAbs(grad_y)
    # grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad


def sample_receptive_fields(image, sampling_mask):
    image = image/255 #scale to 0-1

    active_pixels = torch.from_numpy(image).view(1,image.shape[0],image.shape[1])*sampling_mask
    size_phosphene = sampling_mask.sum(axis=(1,2))
    active_per_phosphene = torch.sum(active_pixels,dim=(1,2))
    # stim = active_per_phosphene/sampling_mask.sum(axis=(1,2))
    stim = torch.zeros_like(active_per_phosphene)
    mask = (size_phosphene!=0)
    stim[mask] = active_per_phosphene[mask] / size_phosphene[mask]
    # stim = (np.einsum('jk, ijk -> i', image, sampling_mask)/sampling_mask.sum(axis=(1,2)))#.clamp(0.,1.) #stim is the fraction of active pixels in a receptive field
    
    stim = 80e-6*stim #DISCUSS: max stimulation?
    stim[(stim>5e-6)&(stim<30e-6)] = 30e-6 #TODO: parameter?
    return stim

def sample_centers(image, pMaps):
    image = image/255 #scale to 0-1

    phosphene_centers = pMaps<0.0001
    print(phosphene_centers.sum())
    active_pixels = torch.from_numpy(image).view(1,image.shape[0],image.shape[1])*phosphene_centers.float()

    active_per_phosphene = torch.amax(active_pixels,dim=(1,2))

    stim = 80e-6*active_per_phosphene

    return stim


