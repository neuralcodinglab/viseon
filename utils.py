import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import datetime
import logging
import torchvision
# import noise
import pandas as pd
import os
import argparse
import yaml

import pickle




# Dilation is used for the reg-loss on the phosphene image: phosphenes do not have to map 1 on 1, small offset is allowed.
def dilation5x5(img, kernel=None):
    if kernel is None:
        kernel = torch.tensor([[[[0., 0., 1., 0., 0.],
                              [0., 1., 1., 1., 0.],
                              [1., 1., 1., 1., 1.],
                              [0., 1., 1., 1., 0.],
                              [0., 0., 1., 0., 0.]]]], requires_grad=False, device=img.device)
    return torch.clamp(torch.nn.functional.conv2d(img, kernel, padding=kernel.shape[-1]//2), 0, 1)

def dilation3x3(img, kernel=None):
    if kernel is None:
        kernel = torch.tensor([[[
                              [ 0, 1., 0.],
                              [ 1., 1., 1.],
                              [ 0., 1., 0.],]]], requires_grad=False, device=img.device)
    return torch.clamp(torch.nn.functional.conv2d(img, kernel, padding=kernel.shape[-1]//2), 0, 1)

def resize(x, out_size=(256,256), interpolation='bilinear'):
    """interpolate/resize tensor to out_size"""
    return torch.nn.functional.interpolate(x, size=out_size, mode=interpolation)

def normalize(x):
    """scale to range [0, 1]"""
    return (x - x.min()) / (x.max()-x.min())

def undo_standardize(x, mean=0.459, std=0.227):
    """maps standardized grayscale images to range [0, 1]"""
    return (x*std+mean).clip(0,1)

def load_config(yaml_file):
    with open(yaml_file) as file:
        raw_content = yaml.load(file,Loader=yaml.FullLoader) # nested dictionary
    return {k:v for params in raw_content.values() for k,v in params.items()} # unpacked


class CustomSummaryTracker():
    """Helper for saving training history, model output, loss, etc.."""
    def __init__(self):
        self.history = dict()

    def get(self):
        return self.history

    def update(self, new_entries):
        for key, value in new_entries.items():
            if key in self.history:
                self.history[key].append(value)
            else:
                self.history[key] = [value]

# For basic plotting of images with labels as title
def plot_images(img_tensor,title=None,classes=None):

    # Un-normalize if images are normalized
    if img_tensor.min()<0:
        if img_tensor.shape[1]==3:
            normalizer = TensorNormalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            normalizer = TensorNormalizer(mean=0.459, std=0.227)
        img_tensor = normalizer.undo(img_tensor)

    # Make numpy
    img = img_tensor.detach().cpu().numpy()


    # Plot all
    for i in range(len(img)):
        plt.subplot(1,len(img),i+1)
        if type(title) is list:
            plt.title(title[i])
        elif title is not None and classes is not None:
            plt.title(classes[title[i].item()])
        if img.shape[1]==1 or len(img.shape)==3 or len(img.shape)==5:
            plt.imshow(np.squeeze(img[i]),cmap='gray',vmin=0,vmax=1)
        elif img.shape[1]==2:
            plt.imshow(img[i][1],cmap='gray',vmin=0,vmax=1)
        else:
            plt.imshow(img[i].transpose(1,2,0))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    return

def log_gradients_in_model(model, model_name, logger, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            logger.add_histogram(f"{model_name}/{tag}", value.grad.cpu(), step)

def save_pickle(data_dict, path):
    """saves dict entries to path, as pickle file"""
    # Make directory if not exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Write model output to pickle
    for name, data in data_dict.items():
        with open(os.path.join(path, f'{name}.pickle'), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# To do (or undo) normalization on torch tensors
class TensorNormalizer(object):
    """To normalize and un-normalize image tensors. For grayscale images uses scalar values for mean and std.
    When called, the  number of channels is automatically inferred."""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std  = std
    def __call__(self,image):
        if image.shape[1]==3:
            return torch.stack([(image[:, c, ...] - self.mean[c]) / self.std[c] for c in range(3)],dim=1)
        else:
            return (image-self.mean)/self.std
    def undo(self,image):
        if image.shape[1]==3:
            return torch.stack([image[:, c, ...]* self.std[c] + self.mean[c] for c in range(3)],dim=1)
        else:
            return image*self.std+self.mean

# To convert to 3-channel format (or reversed)
class RGBConverter(object):
    def __init__(self,weights=[.3,.59,.11]):
        self.weights=weights
        self.copy_channels = torchvision.transforms.Lambda(lambda img:img.repeat(1,3,1,1))
    def __call__(self,image):
        assert len(image.shape) == 4 and image.shape[1] == 1
        image = self.copy_channels(image)
        return image
    def to_gray(self,image):
        assert len(image.shape) == 4 and image.shape[1] == 3
        image = torch.stack([self.weights[c]*image[:,c,:,:] for c in range(3)], dim=1)
        image = torch.sum(image,dim=1,keepdim=True)
        return image
