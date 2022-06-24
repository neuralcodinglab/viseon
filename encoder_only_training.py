import csv
import os
## Import statements

import os
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# Local dependencies
import model,utils
import local_datasets
from simulator import init_a_bit_less

from loss_functions import CustomLoss,Representation_Loss

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader 

from torch.utils.tensorboard import SummaryWriter


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

    use_cuda = False if cfg.device=='cpu' else True

    pMap,sigma_0, activation_mask, threshold, thresh_slope, args = init_a_bit_less(use_cuda=use_cuda)
    models['simulator'] = model.E2E_RealisticPhospheneSimulator(pMap,sigma_0, activation_mask, threshold, thresh_slope, args, device=cfg.device).to(cfg.device)

    # Dataset
    dataset = dict()
    if cfg.dataset == 'characters':
        directory = './datasets/Characters/'
        trainset = local_datasets.Character_Dataset(device=cfg.device,directory=directory)
        valset = local_datasets.Character_Dataset(device=cfg.device,directory=directory,validation = True) 
    elif cfg.dataset == 'ADE20K':
        directory = './datasets/ADE20K/'
        # load_preprocessed = True if os.path.exists(directory+'processed_train_inputs.pkl') else False
        load_preprocessed = True
        trainset = local_datasets.ADE_Dataset(device=cfg.device,directory=directory,load_preprocessed=load_preprocessed)
        valset = local_datasets.ADE_Dataset(device=cfg.device,directory=directory,validation=True,load_preprocessed=load_preprocessed)
        # trainset = local_datasets.ADE_Dataset(device=cfg.device,directory='./datasets/Characters')
        # valset = local_datasets.ADE_Dataset(device=cfg.device,validation=True)
    dataset['trainloader'] = DataLoader(trainset,batch_size=int(cfg.batch_size),shuffle=True)
    dataset['valloader'] = DataLoader(valset,batch_size=int(cfg.batch_size),shuffle=False)

    # Optimization
    optimization = dict()
    if cfg.optimizer == 'adam':
        optimization['encoder'] = torch.optim.Adam(models['encoder'].parameters(),lr=cfg.learning_rate)
    elif cfg.optimizer == 'sgd':
        optimization['encoder'] = torch.optim.SGD(models['encoder'].parameters(),lr=cfg.learning_rate)
    # optimization['lossfunc'] = CustomLoss(recon_loss_type=cfg.reconstruction_loss,
    #                                             recon_loss_param=cfg.reconstruction_loss_param,
    #                                             stimu_loss_type=cfg.sparsity_loss,
    #                                             kappa=cfg.kappa,
    #                                             device=cfg.device)                
    optimization['lossfunc'] = Representation_Loss(loss_type=cfg.representation_loss, loss_param=cfg.representation_loss_param, device=cfg.device)                  
    
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
    


def train(models, dataset, optimization, train_settings, tb_writer):
    
    ## A. Unpack parameters
   
    # Models
    encoder   = models['encoder']
    simulator = models['simulator']
    
    # Dataset
    trainloader = dataset['trainloader']
    valloader   = dataset['valloader']
    
    # Optimization
    encoder_optim = optimization['encoder']
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
        csv_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['epoch','i']+logstats)
    

    ## C. Training Loop
    n_not_improved = 0

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        count_samples=1
        count_val=1
        # count_backwards=1
        logger('Epoch %d' % (epoch+1))
        train_sample_iter = np.random.randint(0,len(trainloader))
        for i, data in enumerate(tqdm(trainloader, desc='Training'), 0):
            image,label = data

            # TRAINING
            encoder.train()
            encoder.zero_grad()

            # 1. Forward pass
            stimulation = encoder(image)
            phosphenes  = simulator(stimulation)

            if i==train_sample_iter: #save for plotting
                fig = utils.full_fig(image[:5],phosphenes[:5])
                fig.show()
                tb_writer.add_figure(f'{model_name}/input & phosphenes/training',fig,epoch * len(trainloader) + i)  

            count_samples+=1
            # 2. Calculate loss
            loss = loss_function(image=image,
                                 phosphenes=phosphenes)
            
            # 3. Backpropagation
            loss.backward()
            encoder_optim.step()

            # VALIDATION
            if i==len(trainloader) or i % log_interval == 0:
                tb_writer.add_histogram(f'{model_name}/stimulation',stimulation,epoch * len(trainloader) + i)
                utils.log_gradients_in_model(encoder, f'{model_name}/encoder', tb_writer, epoch * len(trainloader) + i)
                count_val+=1
                try:
                    sample_iter = np.random.randint(0,len(valloader))
                    for j, data in enumerate(tqdm(valloader, leave=False, position=0, desc='Validation'), 0):
                        image,label = data #next(iter(valloader))
                        # image,label = next(iter(valloader))
                        # print(label)

                        encoder.eval()

                        with torch.no_grad():

                            # 1. Forward pass
                            stimulation = encoder(image)
                            phosphenes  = simulator(stimulation)           

                            if j==sample_iter: #save for plotting
                                sample_img = image
                                sample_phos = phosphenes

                            # 2. Loss
                            _ = loss_function(image=image,
                                                phosphenes=phosphenes,
                                                validation=True)          

                    reset_train = True if i==len(trainloader) else False #reset running losses if at end of loop, else keep going                                                  
                    stats = loss_function.get_stats(reset_training=reset_train,reset_validation=True)
                    tb_writer.add_scalars(f'{model_name}/Loss/validation', {'val_loss':stats['val_loss'][-1]}, epoch * len(trainloader) + i)
                    tb_writer.add_scalars(f'{model_name}/Loss/training', {'tr_loss':stats['tr_loss'][-1]}, epoch * len(trainloader) + i)

                    fig = utils.full_fig(sample_img[:5],sample_phos[:5])
                    fig.show()
                    tb_writer.add_figure(f'{model_name}/input & phosphenes/validation',fig,epoch * len(trainloader) + i)  

                    
                    # 5. Save model (if best)
                    if  np.argmin(stats['val_loss'])+1==len(stats['val_loss']):
                        savepath = os.path.join(savedir,model_name + '_best_encoder.pth' )#'_e%d_encoder.pth' %(epoch))#,i))
                        tqdm.write('Saving to ' + savepath + '...')
                        torch.save(encoder.state_dict(), savepath)
                        
                        n_not_improved = 0
                    else:
                        n_not_improved = n_not_improved + 1
                        tqdm.write('not improved for %5d iterations' % n_not_improved) 
                        if n_not_improved>converg_crit:
                            break

                    # 5. Prepare for next iteration
                    encoder.train()          

                except StopIteration:
                    pass
        if n_not_improved>converg_crit:
            break
    logger('Finished Training')
    tb_writer.close()
    return {'encoder': encoder}, loss_function.stats



if __name__ == '__main__':
    import pandas as pd
    
    args = utils.get_args()
    cfg = pd.Series(vars(args))
    print(cfg)
    models, dataset, optimization, train_settings = initialize_components(cfg)
    writer = SummaryWriter()
    train(models, dataset, optimization, train_settings, writer)

