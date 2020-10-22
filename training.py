import csv
import os
## Import statements

import os
import numpy as np
import matplotlib.pyplot as plt
import math

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
    def __init__(self, recon_loss_type='mse',recon_loss_param=None, stimu_loss_type=None, kappa=0, device='cpu'):
        """Custom loss class for training end-to-end model with a combination of reconstruction loss and sparsity loss
        reconstruction loss type can be either one of: 'mse' (pixel-intensity based), 'vgg' (i.e. perceptual loss/feature loss) 
        or 'boundary' (weighted cross-entropy loss on the output<>semantic boundary labels).
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
            loss_weights = torch.tensor([1-recon_loss_param,recon_loss_param],device=device)
            self.recon_loss = torch.nn.CrossEntropyLoss(weight=loss_weights)
            self.target = 'label'

        # Stimulation loss 
        if stimu_loss_type=='L1':
            self.stimu_loss = lambda x: torch.mean(.5*(x+1)) #converts tanh to sigmoid first
        elif stimu_loss_type == 'L2':
            self.stimu_loss = lambda x: torch.mean((.5*(x+1))**2) #converts tanh to sigmoid first
        elif stimu_loss_type is None:
            self.stimu_loss = None
        self.kappa = kappa if self.stimu_loss is not None else 0
        
        # Output statistics 
        self.stats = {'tr_recon_loss':[],'val_recon_loss': [],'tr_total_loss':[],'val_total_loss':[]}
        if self.stimu_loss is not None:   
            self.stats['tr_stimu_loss']= []
            self.stats['val_stimu_loss']= []
        self.running_loss = {'recon':0,'stimu':0,'total':0}
        self.n_iterations = 0
        
        
    def __call__(self,image,label,stimulation,phosphenes,reconstruction,validation=False):    
        
        # Target
        if self.target == 'image': # Flag for reconstructing input image or target label
            target = image
        elif self.target == 'label':
            target = label
        
        # Calculate loss
        loss_stimu = self.stimu_loss(stimulation) if self.stimu_loss is not None else torch.tensor(0)
        loss_recon = self.recon_loss(reconstruction,target)
        loss_total = (1-self.kappa)*loss_recon + self.kappa*loss_stimu
        
        if not validation:
            # Save running loss and return total loss
            self.running_loss['stimu'] += loss_stimu.item()
            self.running_loss['recon'] += loss_recon.item()
            self.running_loss['total'] += loss_total.item()
            self.n_iterations += 1
            return loss_total
        else:
            # Return train loss (from running loss) and validation loss
            self.stats['val_recon_loss'].append(loss_recon.item())
            self.stats['val_total_loss'].append(loss_total.item())
            self.stats['tr_recon_loss'].append(self.running_loss['recon']/self.n_iterations)
            self.stats['tr_total_loss'].append(self.running_loss['total']/self.n_iterations)
            if self.stimu_loss is not None:
                self.stats['val_stimu_loss'].append(loss_stimu.item())
                self.stats['tr_stimu_loss'].append(self.running_loss['stimu']/self.n_iterations)  
            self.running_loss = {key:0 for key in self.running_loss}
            self.n_iterations = 0
            return self.stats


def initialize_components(cfg):
    """This function returns the required model, dataset and optimization components to initialize training.
    input: <cfg> training configuration (pandas series, or dataframe row)
    returns: dictionaries with the required model components: <models>, <datasets>,<optimization>, <train_settings>
    """

    # Random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Models
    models = dict()
    models['encoder'] = model.E2E_Encoder(in_channels=cfg.input_channels,
                                        binary_stimulation=cfg.binary_stimulation).to(cfg.device)
    models['decoder'] = model.E2E_Decoder(out_channels=cfg.reconstruction_channels,
                                        out_activation=cfg.out_activation).to(cfg.device)
    if cfg.simulation_type == 'regular':
        pMask = utils.get_pMask(jitter_amplitude=0,dropout=False) # phosphene mask with regular mapping
    elif cfg.simulation_type == 'personalized':
        pMask = utils.get_pMask(seed=1,jitter_amplitude=.5,dropout=True,perlin_noise_scale=.4) # pers. phosphene mask
    models['simulator'] = model.E2E_PhospheneSimulator(pMask=pMask.to(cfg.device),
                                                           sigma=1.5,
                                                           intensity=15,
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
    dataset['valloader'] = DataLoader(valset,batch_size=int(cfg.batch_size),shuffle=False)

    # Optimization
    optimization = dict()
    if cfg.optimizer == 'adam':
        optimization['encoder'] = torch.optim.Adam(models['encoder'].parameters(),lr=cfg.learning_rate)
        optimization['decoder'] = torch.optim.Adam(models['decoder'].parameters(),lr=cfg.learning_rate)
    elif cfg.optimizer == 'sgd':
        optimization['encoder'] = torch.optim.SGD(models['encoder'].parameters(),lr=cfg.learning_rate)
        optimization['decoder'] = torch.optim.SGD(models['decoder'].parameters(),lr=cfg.learning_rate)
    optimization['lossfunc'] = CustomLoss(recon_loss_type=cfg.reconstruction_loss,
                                                recon_loss_param=cfg.reconstruction_loss_param,
                                                stimu_loss_type=cfg.sparsity_loss,
                                                kappa=cfg.kappa,
                                                device=cfg.device)                                   
    
    # Additional train settings
    train_settings = dict()
    if not os.path.exists(cfg.savedir):
        os.makedirs(cfg.savedir)
    train_settings['model_name'] = cfg.model_name
    train_settings['savedir']=cfg.savedir
    train_settings['n_epochs'] = cfg.n_epochs
    train_settings['log_interval'] = cfg.log_interval
    train_settings['convergence_criterion'] = cfg.convergence_crit
    return models, dataset, optimization, train_settings
    


def train(models, dataset, optimization, train_settings):
    
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
        os.makedirs(savedir)
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
            if i==len(trainloader) or i % log_interval == (log_interval-1):
                image,label = next(iter(valloader))

                encoder.eval()
                decoder.eval()

                with torch.no_grad():

                    # 1. Forward pass
                    stimulation = encoder(image)
                    phosphenes  = simulator(stimulation)
                    reconstruction = decoder(phosphenes)            

                    # 2. Loss
                    stats = loss_function(image=image,
                                          label=label,
                                          stimulation=encoder.out,
                                          phosphenes=phosphenes,
                                          reconstruction=reconstruction,
                                          validation=True)            
                               
                # 3. Logging
                logstats = ' | '.join('%s : %.3f' %(key,stats[key][-1]) for key in stats) 
                logger('[%d, %5d] %s' %(epoch,i + 1, logstats))
                with open(csvpath, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([epoch,i + 1]+[stats[key][-1] for key in stats])                

                # 4. Visualization
                plt.figure(figsize=(10,10),dpi=50)
                utils.plot_stats(stats)
                plt.figure(figsize=(10,10),dpi=50)
                utils.plot_images(image[:5])
                plt.figure(figsize=(10,10),dpi=50)
                utils.plot_images(phosphenes[:5])
                plt.figure(figsize=(10,10),dpi=50)
                utils.plot_images(reconstruction[:5])
                if len(label.shape)>1:
                    plt.figure(figsize=(10,10),dpi=50)
                    utils.plot_images(label[:5])    
                
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





