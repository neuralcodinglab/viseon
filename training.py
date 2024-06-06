import csv
import os
## Import statements

import os
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

# Local dependencies
import model,utils
import local_datasets

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader 


class CustomLoss(object):
    def __init__(self, recon_loss_type='mse',recon_loss_param=None,
                 stimu_loss_type=None, kappa=0, phosphene_regularization=None, reg_weight=0,phosphene_loss_param=0, device='cpu'):
        """Custom loss class for training end-to-end model with a combination of reconstruction loss 
        and sparsity loss. It automatically keeps track of the loss statistics. 
        reconstruction loss type can be either one of: 'mse' (pixel-intensity based),
        'vgg' (i.e. perceptual loss/feature loss) or 'boundary' (weighted 
        cross-entropy loss on the output<>semantic boundary labels).
        stimulation loss type (i.e. sparsity loss) can be either 'L1', 'L2' or None.
        
          
        """
        
        # Reconstruction loss
        if recon_loss_type == 'mse':
            self.recon_loss = torch.nn.MSELoss()
            self.target = 'image'
        elif recon_loss_type == 'vgg':
            self.feature_extractor = model.VGG_Feature_Extractor(layer_depth=recon_loss_param,device=device)
            self.recon_loss = lambda x,y: torch.nn.functional.mse_loss(self.feature_extractor(x),self.feature_extractor(y))
            self.target = 'image'
        elif recon_loss_type == 'boundary':
#             loss_weights = torch.tensor([1-recon_loss_param,recon_loss_param],device=device)
#             self.recon_loss = torch.nn.CrossEntropyLoss(weight=loss_weights)
            weight = torch.tensor(recon_loss_param,device=device)
            self.recon_loss = lambda x,y,w=weight: torch.nn.functional.binary_cross_entropy(
                x,y,weight=y*w+(1-y)*(1-w))  
            self.target = 'label'

        # Stimulation loss 
        if stimu_loss_type=='L1':
            self.stimu_loss = lambda x: torch.mean(.5*(x+1)) #converts tanh to sigmoid first
        elif stimu_loss_type == 'L2':
            self.stimu_loss = lambda x: torch.mean((.5*(x+1))**2) #converts tanh to sigmoid first
        elif stimu_loss_type is None:
            self.stimu_loss = None
        self.kappa = kappa if self.stimu_loss is not None else 0
        
        # Phosphene regularization
        if phosphene_regularization == 'L1':
            self.reg_loss = lambda stim,ref: ((.5*(stim+1))-ref).abs().mean()
        elif phosphene_regularization == 'L2':
            self.reg_loss = lambda stim,ref: (((.5*(stim+1))-ref)**2).mean()
        elif phosphene_regularization == 'bce':
            weight_p = torch.tensor(phosphene_loss_param,device=device)
            self.reg_loss = lambda stim,ref,w=weight_p: torch.nn.functional.binary_cross_entropy(.5*(stim+1),ref,weight=ref*w+(1-ref)*(1-w))
        elif phosphene_regularization == 'dist':
            self.reg_loss = lambda stim,ref: (stim*ref).mean()
        elif phosphene_regularization is None:
            self.reg_loss = None
        self.reg_weight = reg_weight if phosphene_regularization is not None else 0
        
        # Output statistics 
        self.loss_types = ['recon_loss', 'total_loss']
        if self.stimu_loss is not None:
            self.loss_types.insert(0,'stimu_loss')
        if self.reg_loss is not None:
            self.loss_types.insert(0,'reg_loss')
        
        self.stats = {mode+loss:[] for loss in self.loss_types for mode in ['val_', 'tr_']}
        self.running_loss = {key:0 for key in self.stats}
        self.running_loss.update({'val_img_count':0,'tr_img_count':0})

    def __call__(self,image,label,stimulation,phosphenes,reconstruction,validation=False,reg_target=None):    
        
        # Target
        if self.target == 'image': # Flag for reconstructing input image or target label
            target = image
        elif self.target == 'label':
            target = label
        
        # Calculate loss
        result = dict()
        if self.reg_loss is not None:
#             result['reg_loss'] = self.reg_loss(stimulation,target[:,0,self.reg_mask[0],self.reg_mask[1]])
            result['reg_loss'] = self.reg_loss(stimulation,reg_target)
        else:
            result['reg_loss'] = torch.tensor(0)
        result['stimu_loss'] = self.stimu_loss(stimulation) if self.stimu_loss is not None else torch.tensor(0)
        result['recon_loss'] = self.recon_loss(reconstruction,target)
        result['total_loss'] = (1-self.kappa)*result['recon_loss'] + self.kappa*result['stimu_loss'] + self.reg_weight * result['reg_loss']
        
        # Store the running loss
        mode = 'val_' if validation else 'tr_'
        for type_ in self.loss_types:
            self.running_loss[mode+type_] += result[type_].item() *len(image)
        self.running_loss[mode+'img_count'] += len(image)
        
        return result['total_loss']
        
    def get_stats(self):
        # append runnning loss to stats, reset running loss, and return stats
        for mode in ['val_','tr_']:
            for type_ in self.loss_types: 
                img_count = self.running_loss[mode+'img_count']
                if img_count != 0: #assert that running loss is not 'empty'
                    self.stats[mode+type_].append(self.running_loss[mode+type_]/img_count) 
                    self.running_loss[mode+type_] = 0
        self.running_loss.update({'val_img_count':0,'tr_img_count':0}) # reset count of loss iterations 
        return self.stats


def initialize_components(cfg):
    """This function returns the required model, dataset and optimization components to initialize training.
    input: <cfg> training configuration (pandas series, or dataframe row)
    returns: dictionaries with the required model components: <models>, <datasets>,<optimization>, <train_settings>
    """

    # Random seed
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Models
    models = dict()
    models['encoder'] = model.E2E_Encoder(in_channels=cfg.input_channels,
                                        binary_stimulation=cfg.binary_stimulation).to(cfg.device)
    models['decoder'] = model.E2E_Decoder(out_channels=cfg.reconstruction_channels,
                                        out_activation=cfg.out_activation).to(cfg.device)
    
    if cfg.simulation_type == 'regular':
        pMask = utils.get_pMask(jitter_amplitude=0,dropout=False,seed=seed) # phosphene mask with regular mapping
    elif cfg.simulation_type == 'irregular':
        pMask      = utils.get_pMask(intensity_var=1, jitter_amplitude=0.5,dropout=False,seed=8)
    models['simulator'] = model.E2E_PhospheneSimulator(pMask=pMask.to(cfg.device),
                                                           sigma=1.5,
                                                           intensity=10,
                                                           device=cfg.device).to(cfg.device)

    # Dataset
    dataset = dict()
    if cfg.dataset == 'characters':
        trainset = local_datasets.Character_Dataset(device=cfg.device)
        valset = local_datasets.Character_Dataset(device=cfg.device,validation = True) 
    elif cfg.dataset == 'ADE20K':
        trainset = local_datasets.ADE_Dataset(device=cfg.device)
        valset = local_datasets.ADE_Dataset(device=cfg.device,validation=True)
    dataset['trainloader'] = DataLoader(trainset,batch_size=int(cfg.batch_size),shuffle=True)
    dataset['valloader'] = DataLoader(valset,batch_size=min([len(valset),100]),shuffle=False)
    dataset['trainset'] = trainset
    dataset['valset'] = valset

    # Optimization
    optimization = dict()
    if cfg.optimizer == 'adam':
        optimization['encoder'] = torch.optim.Adam(models['encoder'].parameters(),lr=cfg.learning_rate)
        optimization['decoder'] = torch.optim.Adam(models['decoder'].parameters(),lr=cfg.learning_rate)
    elif cfg.optimizer == 'sgd':
        optimization['encoder'] = torch.optim.SGD(models['encoder'].parameters(),lr=cfg.learning_rate)
        optimization['decoder'] = torch.optim.SGD(models['decoder'].parameters(),lr=cfg.learning_rate)
    
#     # for regularizing phosphene mapping
#     pMap = torch.nn.functional.interpolate(models['simulator'].pMap.unsqueeze(dim=1),size=(128,128))
#     pLocs = pMap.view(N_PHOSPHENES,-1).argmax(dim=-1)
#     pLocs = pLocs // 128, pLocs % 128 # y and x coordinates of the center of each phosphene
    
    optimization['lossfunc'] = CustomLoss(recon_loss_type=cfg.reconstruction_loss,
                                                recon_loss_param=cfg.reconstruction_loss_param,
                                                stimu_loss_type=cfg.sparsity_loss,
                                                kappa=cfg.kappa,
                                                phosphene_regularization=cfg.phosphene_regularization,
                                                reg_weight=cfg.reg_weight,                                                
                                                device=cfg.device,
                                                phosphene_loss_param=cfg.phosphene_loss_param)                                   
    
    # Additional train settings
    train_settings = dict()
    train_settings['model_name'] = cfg.model_name
    train_settings['savedir']=cfg.savedir
    train_settings['n_epochs'] = cfg.n_epochs
    train_settings['log_interval'] = cfg.log_interval
    train_settings['convergence_criterion'] = cfg.convergence_crit
    return models, dataset, optimization, train_settings
    


def train(models, dataset, optimization, train_settings, visualize=False):
    
    ## A. Unpack parameters
   
    # Models
    encoder   = models['encoder']
    decoder   = models['decoder']
    simulator = models['simulator']
    
    # Dataset
    trainloader = dataset['trainloader']
    valloader   = dataset['valloader']
    
    # Optimization
    encoder_optim = optimization['encoder']
    decoder_optim = optimization['decoder']
    loss_function = optimization['lossfunc']

    
    # Train settings
    model_name   = train_settings['model_name']
    savedir      = train_settings['savedir']
    n_epochs     = train_settings.get('n_epochs',2)
    log_interval = train_settings.get('log_interval',10)
    converg_crit = train_settings.get('convergence_criterion',50)
    
    
    ## B. Logging
    if not os.path.exists(savedir):
        os.makedirs(os.path.join(savedir,'figures'))
    logger = utils.Logger(os.path.join(savedir,'out.log'))
    csvpath = os.path.join(savedir,model_name+'_train_stats.csv')
    logstats = list(loss_function.stats.keys())
    with open(csvpath, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['epoch','i']+logstats)
    

    ## C. Training Loop
    n_not_improved = 0
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        logger('Epoch %d' % (epoch+1))

        for i, data in enumerate(trainloader, 0):
            image,label = data

            # TRAINING
            encoder.train()
            decoder.train()
            encoder.zero_grad()
            decoder.zero_grad()

            # 1. Forward pass
            stimulation = encoder(image)
            phosphenes  = simulator(stimulation)
            reconstruction = decoder(phosphenes)

            # 2. Calculate loss
            loss = loss_function(image=image,
                                 label=label,
                                 stimulation=encoder.out,
                                 phosphenes=phosphenes,
                                 reconstruction=reconstruction)

            # 3. Backpropagation
            loss.backward()
            encoder_optim.step()
            decoder_optim.step()
            

            

            # VALIDATION
            if i==len(trainloader)-1 or i % log_interval == (log_interval-1):


                encoder.eval()
                decoder.eval()
                for _, data in enumerate(valloader):
                    image,label = data
                    with torch.no_grad():

                        # 1. Forward pass
                        stimulation = encoder(image)
                        phosphenes  = simulator(stimulation)
                        reconstruction = decoder(phosphenes)            

                        # 2. Loss
                        loss = loss_function(image=image,
                                              label=label,
                                              stimulation=encoder.out,
                                              phosphenes=phosphenes,
                                              reconstruction=reconstruction,
                                              validation=True)
                stats = loss_function.get_stats()
                               
                # 3. Logging
                logstats = ' | '.join('%s : %.3f' %(key,stats[key][-1]) for key in stats if stats[key]) 
                logger('[%d, %5d] %s' %(epoch,i + 1, logstats))
                with open(csvpath, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([epoch,i + 1]+[stats[key][-1] if stats[key] else 'NaN' for key in stats])                

                # 4. Visualization
                if visualize:
                    results = torch.cat([image[:5,-1:],reconstruction[:5]],axis=-1)
                    results = torch.cat([results,phosphenes[:5]],axis=-2)
                    
                    plt.figure(figsize=(10,10),dpi=50)
                    utils.plot_stats(stats,save_as=os.path.join(savedir,
                                                'figures/_{}_Trainstats.png'.format(model_name)))
                    plt.figure(figsize=(10,3),dpi=100)
                    fn = 'figures/{}_E{:02d}-{:03d}.png'.format(model_name,epoch,i)
                    utils.plot_images(results,save_as=os.path.join(savedir,fn))
                   
                   
                # 5. Save model (if best)
                if  np.argmin(stats['val_total_loss'])+1==len(stats['val_total_loss']):
                    savepath = os.path.join(savedir,model_name + '_best_encoder.pth' )#'_e%d_encoder.pth' %(epoch))#,i))
                    logger('Saving to ' + savepath + '...')
                    torch.save(encoder.state_dict(), savepath)

                    savepath = os.path.join(savedir,model_name + '_best_decoder.pth' )#'_e%d_decoder.pth' %(epoch))#,i))
                    logger('Saving to ' + savepath + '...')
                    torch.save(decoder.state_dict(), savepath)
                    
                    n_not_improved = 0
                else:
                    n_not_improved = n_not_improved + 1
                    logger('not improved for %5d iterations' % n_not_improved) 
                    if n_not_improved>converg_crit:
                        break

                # 5. Prepare for next iteration
                encoder.train()
                decoder.train()            

        if n_not_improved>converg_crit:
            break
    logger('Finished Training')
    
    return {'encoder': encoder, 'decoder':decoder}, loss_function.stats



if __name__ == '__main__':
    import argparse
    import pandas as pd
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-csv","--use_csv", type=str, default=None,
                    help="path/to/csv-file with multiple train configurations")
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
    ap.add_argument("-reg_type", "--phosphene_regularization", type=str, default=None,
                    help="Phosphene regularization mode. Choose from: 'L1', 'L2', 'dist', 'bce'")  
    ap.add_argument("-reg_wgt", "--reg_weight", type=float, default=None,
                    help="Phosphene regularization weight")  
    ap.add_argument("-phl", "--phosphene_loss_param", type=float, default=None,
                    help="Phosphene regularization weight")  

    args = vars(ap.parse_args())
    if args['use_csv'] is None:
        cfg = pd.Series(args)       
        print(cfg)
        models, dataset, optimization, train_settings = initialize_components(cfg)
        train(models, dataset, optimization, train_settings)
    else:
        training_protocol = pd.read_csv(args['use_csv']).replace({'None':None,'True':True,'False':False})
        for i,cfg in training_protocol.iterrows():
            print(cfg)
            models, dataset, optimization, train_settings = initialize_components(cfg)
            train(models, dataset, optimization, train_settings)

