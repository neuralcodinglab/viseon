import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torchvision
import torch
import utils,training,model
import local_datasets
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader 
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio


def read_train_stats(filename, delimiter=','):
    """Reads the csv file with training statistics into pandas DataFrame and returns
    these in 'melted' format, which is useful for plotting the training curves. Also 
    returns the best-performance training statistics"""
    # Read train statistics and best-performance results 
    stats = pd.read_csv(filename, delimiter=delimiter) # os.path.join(cfg.savedir,cfg.model_name+'_train_stats.csv')
    stats['epoch'] = stats.epoch + (stats.i / stats.i.max())
    best_result = stats.iloc[stats['val_total_loss'].idxmin()]
    
    # Convert to melted format
    value_vars=[col for col in stats.columns if col.endswith('loss')]
    train_stats = pd.melt(stats,id_vars=['epoch'],value_vars=value_vars)
    return train_stats, best_result        

def segmentation_metrics(pred, label):
    """ Calculate sensitivity, specificity, precision and accuray from softmax 
    prediction (tensor) and ground truth target (tensor)"""
    # Prediction
    pred = pred.argmax(axis=1)

    # Confusion quadrants
    tp = pred[label==1].sum().float()
    fp = pred[label==0].sum().float()
    fn = label[pred==0].sum().float()
    tn = (label[pred==0] == 0).sum().float()

    # Performance metrics 
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    prec = tp/(tp+fp)
    acc  = (tp + tn) / label.numel()
    return pd.Series({'sensitivity': sens, 'specificity': spec, 'precision': prec, 'accuracy': acc})


def evaluate_saved_model(cfg, visualize=None):
    """ loads the saved model parametes for given configuration <cfg> and returns the performance
    metrics on the validation dataset. The <visualize> argument can be set equal to any positive
    integer that represents the amount of example figures to plot."""
    # Load configurations
    models, dataset, optimization, train_settings = training.initialize_components(cfg)
    encoder = models['encoder']
    decoder = models['decoder']
    simulator = models['simulator']
    valloader = dataset['valloader']
    lossfunc = optimization['lossfunc']
    
    # Load model parameters
    encoder.load_state_dict(torch.load(os.path.join(cfg.savedir,cfg.model_name+'_best_encoder.pth')))
    decoder.load_state_dict(torch.load(os.path.join(cfg.savedir,cfg.model_name+'_best_decoder.pth')))
    encoder.eval()
    decoder.eval()
    
    # Forward pass (validation set)
    image,label = next(iter(valloader))
    with torch.no_grad():
        stimulation = encoder(image)
        phosphenes  = simulator(stimulation)
        reconstruction = decoder(phosphenes)    
    
    
    # Visualize results
    if visualize is not None:
        n_figs = 4 if cfg.reconstruction_loss == 'boundary' else 3
        n_examples = visualize
        plt.figure(figsize=(n_figs,n_examples),dpi=200)

        for i in range(n_examples):
            plt.subplot(n_examples,n_figs,n_figs*i+1)
            plt.imshow(image[i].squeeze().cpu().numpy(),cmap='gray')
            plt.axis('off')
            plt.subplot(n_examples,n_figs,n_figs*i+2)
            plt.imshow(phosphenes[i].squeeze().cpu().numpy(),cmap='gray')
            plt.axis('off')
            plt.subplot(n_examples,n_figs,n_figs*i+3)
            plt.imshow(reconstruction[i].squeeze().cpu().numpy(),cmap='gray')
            plt.axis('off')
            if n_figs > 3:
                plt.subplot(n_examples,n_figs,n_figs*i+4)
                plt.imshow(label[i].squeeze().cpu().numpy(),cmap='gray')
                plt.axis('off')
        plt.show()
    
    # Calculate performance metrics
    im_pairs = [[im.squeeze().cpu().numpy(),trg.squeeze().cpu().numpy()] for im,trg in zip(image,reconstruction)]
    
    if cfg.reconstruction_loss == 'boundary':
        metrics=pd.Series() #TODO
    else:
        mse = [mean_squared_error(*pair) for pair in im_pairs]
        ssim = [structural_similarity(*pair, gaussian_weigths=True) for pair in im_pairs]
        psnr = [peak_signal_noise_ratio(*pair) for pair in im_pairs]
        metrics=pd.Series({'mse':np.mean(mse),
                           'ssim':np.mean(ssim),
                           'psnr':np.mean(psnr)})
    return metrics

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow 
    (see: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063)'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)],
               ['max-gradient', 'mean-gradient', 'zero-gradient'])