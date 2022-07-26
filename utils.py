import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import datetime
import logging
import torchvision
import noise
import pandas as pd
import os 
import argparse

def pred_to_intensities(intensitiesArray, predictions):
    # activation_values = predictions * (intensitiesArray.max()-intensitiesArray.min())+intensitiesArray.min()
    # print(f"activation_values scaled: {activation_values.mean(), activation_values.max()}")
    activation_values = predictions
    intensitiesArray = torch.tile(intensitiesArray,(predictions.shape[1],1))
    activation_values = intensitiesArray[torch.arange(intensitiesArray.shape[0]),(torch.abs(intensitiesArray - activation_values.unsqueeze(2))).argmin(axis=2)]
    # print(f"activation_values binned: {activation_values.mean(), activation_values.max()}")
    return activation_values

class Logger(object):
    def __init__(self,log_file='out.log'):
        self.logger = logging.getLogger()
        hdlr = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr) 
        self.logger.setLevel(logging.INFO)
    def __call__(self,message):
        #outputs to Jupyter console
        print('{} {}'.format(datetime.datetime.now(), message))
        #outputs to file
        self.logger.info(message)

def get_args(args_list=None):
    ap = argparse.ArgumentParser(fromfile_prefix_chars='@')

    def none_or_str(value):
        if value == 'None':
            return None
        return value

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    ap.add_argument("-m", "--model_name", type=str, default="demo_model",
                    help="model name")
    ap.add_argument("-dir", "--savedir", type=str, default="./out/demo",
                    help="directory for saving the model parameters and training statistics")
    ap.add_argument("-s", "--seed", type=int, default=0,
                    help="seed for random initialization")
    ap.add_argument("-e", "--n_epochs", type=int, default=3,
                    help="number of training epochs")   
    ap.add_argument("-l", "--log_interval", type=int, default=10,
                    help="number of batches after which to evaluate model (and logged)")   
    ap.add_argument("-crit", "--convergence_crit", type=int, default=30,
                    help="stop-criterion for convergence: number of evaluations after which model is not improved")   
    ap.add_argument("-bin", "--binary_stimulation", type=str2bool, default=True,
                    help="use quantized (binary) instead of continuous stimulation protocol")
    ap.add_argument("-bs", "--binned_stimulation", type=str2bool, default=True,
                    help="use binned instead of continuous stimulation protocol")
    ap.add_argument("-sim", "--simulation_type", type=str, default="regular",
                    help="'regular' or 'personalized' phosphene mapping") 
    ap.add_argument("-in", "--input_channels", type=int, default=1,
                    help="only grayscale (single channel) images are supported for now")   
    ap.add_argument("-out", "--reconstruction_channels", type=int, default=1,
                    help="only grayscale (single channel) images are supported for now")     
    ap.add_argument("-act", "--out_activation", type=str, default="sigmoid",
                    help="use 'sigmoid' for grayscale reconstructions, 'softmax' for boundary segmentation task")   
    ap.add_argument("-d", "--dataset", type=str, default="characters",
                    help="'charaters' dataset and 'ADE20K' are supported")   
    ap.add_argument("-dev", "--device", type=str, default="cuda:0",
                    help="e.g. use 'cpu' or 'cuda:0' ")   
    ap.add_argument("-n", "--batch_size", type=int, default=30,
                    help="'charaters' dataset and 'ADE20K' are supported")   
    ap.add_argument("-opt", "--optimizer", type=str, default="adam",
                    help="only 'adam' is supported for now")   
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.0001,
                    help="Use higher learning rates for VGG-loss (perceptual reconstruction task)")  
    ap.add_argument("-rl", "--reconstruction_loss", type=str, default='mse',
                    help="'mse', 'VGG' or 'boundary' loss are supported ") 
    ap.add_argument("-p", "--reconstruction_loss_param", type=float, default=0,
                    help="In perceptual condition: the VGG layer depth, boundary segmentation: cross-entropy class weight") 
    ap.add_argument("-L", "--sparsity_loss", type=none_or_str, default='L1',
                    help="choose L1 or L2 type of sparsity loss (MSE or L1('taxidrivers') norm)") 
    ap.add_argument("-repl", "--representation_loss", type=none_or_str, default=None,
                    help="loss term between phosphene representation and input") 
    ap.add_argument("-pr", "--representation_loss_param", type=float, default=0,
                    help="vgg layer depth for representation loss")                
    ap.add_argument("-k", "--kappa", type=float, default=0.01,
                    help="sparsity weight parameter kappa")    

    if args_list is not None:
        return ap.parse_args(args_list)
    else:
        return ap.parse_args()
        
        
def get_pMask(size=(256,256),phosphene_density=32,seed=1,jitter_amplitude=0,dropout=False,perlin_noise_scale=.4):

    # Define resolution and phosphene_density
    [nx,ny] = size
    n_phosphenes = phosphene_density**2 # e.g. n_phosphenes = 32 x 32 = 1024
    pMask = torch.zeros(size)


    # Custom 'dropout_map'
    p_dropout = perlin_noise_map(shape=size,scale=perlin_noise_scale*size[0],seed=seed)
    np.random.seed(seed)

    for p in range(n_phosphenes):
        i, j = divmod(p, phosphene_density)

        jitter = np.round(np.multiply(np.array([nx,ny])//phosphene_density, jitter_amplitude * (np.random.rand(2)-.5))).astype(int)
        rx = (j*nx//phosphene_density) + nx//(2*phosphene_density) + jitter[0]
        ry = (i*ny//phosphene_density) + ny//(2*phosphene_density) + jitter[1]

        rx = np.clip(rx,0,nx-1)
        ry = np.clip(ry,0,ny-1)
        if dropout==True:
            pMask[rx,ry] = np.random.choice([0.,1.], p=[p_dropout[rx,ry],1-p_dropout[rx,ry]])
        else:
            pMask[rx,ry] = 1.
            
    return pMask       
 

def perlin_noise_map(seed=0,shape=(256,256),scale=100,octaves=6,persistence=.5,lacunarity=2.):
    out = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out[i][j] = noise.pnoise2(i/scale, 
                                        j/scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=shape[0], 
                                        repeaty=shape[1], 
                                        base=seed)
    out = (out-out.min())/(out.max()-out.min())
    return out

def plot_stats(stats):
    """ Plot dict containing lists of train statistics"""
    for key in stats:
        plt.plot(stats[key], label=key)
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('training statistics')
    plt.show()
    return

def full_fig(*tensors,title=None,classes=None): #img_tensor, phs_tensor, recon_tensor
    # tensors = [img_tensor,phs_tensor,recon_tensor]
    # images = np.zeros((3,img_tensor.shape[0],img_tensor.shape[1],img_tensor.shape[2],img_tensor.shape[3]))
    fig = plt.figure(figsize=(5*len(tensors),5*tensors[0].shape[0]))

    for i,t in enumerate(tensors):
        # if t.min()<0:
        #     if t.shape[1]==3:
        #         normalizer = TensorNormalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     elif img_tensor.shape[1]==1:
        #         normalizer = TensorNormalizer(mean=0.459, std=0.227)
        #     t = normalizer.undo(t)
        
        # Make numpy
        img = t.detach().cpu().numpy()

        for j in range(img.shape[0]):
            ax = fig.add_subplot(len(tensors),img.shape[0],i*img.shape[0]+(j+1))

            if type(title) is list:
                fig.title(title[j])
            elif title is not None and classes is not None:
                fig.title(classes[title[j].item()])
            if img.shape[1]==1 or len(img.shape)==3:
                im = ax.imshow(np.squeeze(img[j]),cmap='gray',vmin=0,vmax=1)
            elif img.shape[1]==2:    
                im = ax.imshow(img[j][1],cmap='gray',vmin=0,vmax=1)
            else:
                im = ax.imshow(img[j].transpose(1,2,0))
            ax.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        fig.tight_layout(h_pad=1)
    return fig

# For basic plotting of images with labels as title
def plot_images(img_tensor,title=None,classes=None):
    
    # Un-normalize if images are normalized  
    if img_tensor.min()<0:
        if img_tensor.shape[1]==3:
            normalizer = TensorNormalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif img_tensor.shape[1]==1:
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
        if img.shape[1]==1 or len(img.shape)==3:
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

# To do (or undo) normalization on torch tensors
class TensorNormalizer(object):
    """To normalize and un-normalize image tensors. For grayscale images uses scalar values for mean and std.
    When called, the  number of channels is automatically inferred."""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std  = std
    def __call__(self,image):
        if image.shape[1]==3:
            return torch.stack([(image[:, c, :, :] - self.mean[c]) / self.std[c] for c in range(3)],dim=1)
        else:
            return (image-self.mean)/self.std
    def undo(self,image):
        if image.shape[1]==3:
            return torch.stack([image[:, c, :, :]* self.std[c] + self.mean[c] for c in range(3)],dim=1)
        else:
            return image*self.std+self.mean

def add_noise(clean_image, level=0.3):
    """Inverts random elements of the original image
    value of noise level should be chosen in range [0. , 0.5]"""
    # Add noise (random inversion of the image)
    image = clean_image.clone()
    mask = np.random.randint(0,image.numel(),int(level*image.numel()))
    image.flatten()[mask] = 1-image.flatten()[mask]
    return image

    
    
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
    
