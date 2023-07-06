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
import image_similarity_measures
from image_similarity_measures.quality_metrics import fsim as feature_similarity
from sklearn.metrics import roc_auc_score

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

def evaluate_saved_model(cfg, visualize=None, savefig=False, seed=0):
    """ loads the saved model parametes for given configuration <cfg> and returns the performance
    metrics on the validation dataset. The <visualize> argument can used for passing a list of 
    integers that represent example images to plot."""
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
    
    n_processed = 0
    metrics = None
    BATCH_SIZE = 100 # Default batch size is 100 for evaluation 
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if visualize is not None:
        n_figs = 4 if cfg.reconstruction_loss == 'boundary' else 3
        n_examples = len(visualize)
        plt.figure(figsize=(n_figs,n_examples),dpi=300)
        i = 0
        
    # Forward pass (validation set)
    for batch, data in enumerate(valloader, 0):
        image,label = data
        with torch.no_grad():
            stimulation = encoder(image)
            phosphenes  = simulator(stimulation)
            reconstruction = decoder(phosphenes)
        
        # Undo normalization (if necessary)
        if image.min()<0:
            if image.shape[1]==3:
                normalizer = utils.TensorNormalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            elif image.shape[1]==1:
                normalizer = utils.TensorNormalizer(mean=0.459, std=0.227)
            image = normalizer.undo(image)
            reconstruction = normalizer.undo(reconstruction) if lossfunc.target == 'image' else reconstruction
        
        # Calculate the performance metrics (perc. active electrodes, IQA, seqm. performance)
        pred = reconstruction
        targ = image if lossfunc.target == 'image' else label
        
        if metrics is None:
            # calculate performance metrics
            metrics = performance_metrics(stimulation, pred, targ, targ_type=lossfunc.target)
            n_processed = len(image)
        else:
            # update with new metrics
            new_metrics = performance_metrics(stimulation, pred, targ, targ_type=lossfunc.target)
            metrics = (metrics*n_processed + new_metrics*len(image))/(n_processed + len(image)) #avg. over all metrics
            n_processed = n_processed + len(image)

    
        # Visualize results
        if visualize is not None:
            for example in visualize:
                b,idx = divmod(example, BATCH_SIZE)
                if b == batch:
                    plt.subplot(n_examples,n_figs,n_figs*i+1)
                    plt.imshow(image[idx].squeeze().cpu().numpy(),cmap='gray')
                    plt.axis('off')
                    plt.subplot(n_examples,n_figs,n_figs*i+2)
                    plt.imshow(phosphenes[idx].squeeze().cpu().numpy(),cmap='gray')
                    plt.axis('off')

                    if n_figs > 3:
                        plt.subplot(n_examples,n_figs,n_figs*i+3)
                        plt.imshow(reconstruction[idx].squeeze().cpu().numpy(),cmap='gray')
                        plt.axis('off')
                        plt.subplot(n_examples,n_figs,n_figs*i+4)
                        plt.imshow(label[idx].squeeze().cpu().numpy(),cmap='gray')
                        plt.axis('off')

                    else:
                        plt.subplot(n_examples,n_figs,n_figs*i+3)
                        plt.imshow(reconstruction[idx].squeeze().cpu().numpy(),cmap='gray')
                        plt.axis('off')
                    i += 1
            if batch == len(valloader)-1:
                if savefig:
                    plt.savefig(os.path.join(cfg.savedir,cfg.model_name+'eval.png'))

                plt.tight_layout()
                plt.show()
    return metrics
            
    
    
def performance_metrics(stimulation, pred, targ, targ_type='image'):
    """ Calculate electrode activation and image quality metrics or
    segmentation performance metrics """

    
    # 1. Percentage of activated electrodes
    perc_active = np.mean([100.*(s>.5).sum().item() / s.numel() for s in stimulation])
        
    if targ_type == 'image':    
        # 2. Image quality assessment metrics 
        im_pairs = [[prd.squeeze().cpu().numpy(),trg.squeeze().cpu().numpy()] for prd,trg in zip(pred,targ)]
        fsim = np.mean([feature_similarity(prd[:,:,None],trg[:,:,None]) for prd,trg in im_pairs])
        mse  = np.mean([mean_squared_error(*pair) for pair in im_pairs])
        ssim = np.mean([structural_similarity(*pair, gaussian_weigths=True) for pair in im_pairs])
        psnr = np.mean([peak_signal_noise_ratio(*pair) for pair in im_pairs])
        
        return pd.Series({'perc_active': perc_active,
                        'mse': mse,
                        'ssim':ssim,
                        'fsim':fsim,
                        'psnr':psnr})

    else: # targ_type == 'label'
        # 3. Segmentation performance metrics
        
        # Area under the ROC-curve (raw predictions, fp-rate vs tp-rate)
        auc  = roc_auc_score(targ.flatten().cpu().numpy(),pred.flatten().cpu().numpy())
        
        # Thresholded predictions
        pred = (pred>0.5).float()

        # Confusion quadrants
        tp = pred[targ==1].sum().float()
        fp = pred[targ==0].sum().float()
        fn = targ[pred==0].sum().float()
        tn = (targ[pred==0] == 0).sum().float()

        # Performance metrics 
        sens = (tp/(tp+fn)).cpu().numpy()
        spec = (tn/(tn+fp)).cpu().numpy()
        prec = (tp/(tp+fp)).cpu().numpy()
        acc  = ((tp + tn) / targ.numel()).cpu().numpy()
        
    
        return pd.Series({'perc_active': perc_active,
                            'sensitivity': sens,
                            'specificity': spec,
                            'precision': prec,
                            'accuracy': acc,
                            'auc_score': auc})



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
    
    
if __name__ == '__main__':
    import argparse
    import pandas as pd
    
    ap = argparse.ArgumentParser()
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
    ap.add_argument("-bin", "--binary_stimulation", type=bool, default=True,
                    help="use quantized (binary) instead of continuous stimulation protocol")   
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
                    help="only 'adam' is supporte for now")   
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.0001,
                    help="Use higher learning rates for VGG-loss (perceptual reconstruction task)")  
    ap.add_argument("-rl", "--reconstruction_loss", type=str, default='mse',
                    help="'mse', 'VGG' or 'boundary' loss are supported ") 
    ap.add_argument("-p", "--reconstruction_loss_param", type=float, default=0,
                    help="In perceptual condition: the VGG layer depth, boundary segmentation: cross-entropy class weight") 
    ap.add_argument("-L", "--sparsity_loss", type=str, default='L1',
                    help="choose L1 or L2 type of sparsity loss (MSE or L1('taxidrivers') norm)") 
    ap.add_argument("-k", "--kappa", type=float, default=0.01,
                    help="sparsity weight parameter kappa")    

    cfg = pd.Series(vars(ap.parse_args()))
    metrics = evaluate_saved_model(cfg, 5, savefig=True)
    print(metrics)