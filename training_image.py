## Import statements
import csv
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local dependencies
import model,utils
import local_datasets
from simulator.simulator import GaussianSimulator
from simulator.init import init_probabilistically
from simulator.utils import load_params

from loss_functions import CustomLoss

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
    models['decoder'] = model.E2E_Decoder(out_channels=cfg.reconstruction_channels,
                                        out_activation=cfg.out_activation).to(cfg.device)

    # use_cuda = False if cfg.device=='cpu' else True
    if cfg.simulation_type == 'realistic':
        params = load_params('simulator/config/params.yaml')
        r, phi = init_probabilistically(params,n_phosphenes=1024)
        models['simulator'] = model.E2E_RealisticPhospheneSimulator(cfg,params, r, phi).to(cfg.device)
    else:
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
        directory = './datasets/Characters/'
        trainset = local_datasets.Character_Dataset(device=cfg.device,directory=directory)
        valset = local_datasets.Character_Dataset(device=cfg.device,directory=directory,validation = True) 
    elif cfg.dataset == 'ADE20K':
        directory = './datasets/ADE20K/'
        load_preprocessed = True if os.path.exists(directory+'/images/processed_train') and os.path.exists(directory+'/images/processed_val') else False
        # load_preprocessed = True
        trainset = local_datasets.ADE_Dataset(device=cfg.device,directory=directory,load_preprocessed=load_preprocessed, circular_mask=True,normalize=False)
        valset = local_datasets.ADE_Dataset(device=cfg.device,directory=directory,validation=True,load_preprocessed=load_preprocessed, circular_mask=True, normalize=False)
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
                                                phosrep_loss_type=cfg.representation_loss,
                                                phosrep_loss_param=cfg.representation_loss_param,
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
    train_settings['binned_stimulation'] = cfg.binned_stimulation
    if cfg.binned_stimulation:
        models['intensities_array'] = torch.linspace(0,params['encoding']['max_stim'],params['encoding']['n_config']+1,device=cfg.device)
    return models, dataset, optimization, train_settings
    


def train(models, dataset, optimization, train_settings, tb_writer):
    
    ## A. Unpack parameters
   
    # Models
    encoder   = models['encoder']
    decoder   = models['decoder']
    simulator = models['simulator']
    
    if train_settings['binned_stimulation']:
        intensities_array = models['intensities_array']

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
        csv_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['epoch','i']+logstats)
    

    ## C. Training Loop
    n_not_improved = 0
    running_loss = 0.0
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        count_samples=1
        count_val=1
        # count_backwards=1
        logger('Epoch %d' % (epoch+1))

        for i, data in enumerate(tqdm(trainloader, desc='Training'), 0):
            image,label = data
            # print(np.unique(label.cpu()))
            # print('in training loop')
            # TRAINING
            encoder.train()
            decoder.train()
            encoder.zero_grad()
            decoder.zero_grad()

            # 1. Forward pass
            stimulation = encoder(image)
            # print(f"stimulation: {stimulation.shape}")
            # proxy_stim = 30*torch.ones_like(stimulation)
            phosphenes  = simulator(stimulation)
            # print(f"phosphenes: {phosphenes.shape}")
            reconstruction = decoder(phosphenes)
            # print(f"reconstruction: {reconstruction.shape}")
            # print(f"passed sample through models {count_samples} times")
            count_samples+=1
            # 2. Calculate loss
            loss = loss_function(image=image,
                                 label=label,
                                 stimulation=encoder.out,
                                 phosphenes=phosphenes,
                                 reconstruction=reconstruction)
            # print("calculated loss")
            
            # 3. Backpropagation
            loss.backward()
            
            # print("backward step")
            encoder_optim.step()
            decoder_optim.step()
            # print("optimizer step")
            # print(f"optimizer step {count_backwards}")
            # count_backwards+=1
            del loss
            # running_loss += loss.item()

            # VALIDATION
            if i==len(trainloader) or i % log_interval == 0:
                # print("Running validation loop")
                # tb_writer.add_scalar('Loss/train', running_loss/log_interval, epoch * len(trainloader) + i)
                # running_loss = 0.0
                tb_writer.add_histogram(f'{model_name}/stimulation',stimulation,epoch * len(trainloader) + i)
                utils.log_gradients_in_model(encoder, f'{model_name}/encoder', tb_writer, epoch * len(trainloader) + i)
                utils.log_gradients_in_model(decoder, f'{model_name}/decoder', tb_writer, epoch * len(trainloader) + i)
                # print(stimulation)
                count_val+=1
                try:
                    sample_iter = np.random.randint(0,len(valloader))
                    for j, data in enumerate(tqdm(valloader, leave=False, position=0, desc='Validation'), 0):
                        image,label = data #next(iter(valloader))
                        # print(label)
                            
                        encoder.eval()
                        decoder.eval()

                        with torch.no_grad():

                            # 1. Forward pass
                            stimulation = encoder(image)
                            if train_settings['binned_stimulation']:
                                stimulation = utils.pred_to_intensities(intensities_array, stimulation)
                            phosphenes  = simulator(stimulation)
                            reconstruction = decoder(phosphenes)   

                            if j==sample_iter: #save for plotting
                                sample_img = image
                                sample_phos = phosphenes
                                sample_recon = reconstruction

                            # 2. Loss
                            _ = loss_function(image=image,
                                                label=label,
                                                stimulation=encoder.out,
                                                phosphenes=phosphenes,
                                                reconstruction=reconstruction,
                                                validation=True) 

                    reset_train = True if i==len(trainloader) else False #reset running losses if at end of loop, else keep going
                    stats = loss_function.get_stats(reset_train,reset_val=True) #reset val loss always after validation loop completes

                    tb_writer.add_scalars(f'{model_name}/Loss/validation', {key: stats[key][-1] for key in ['val_recon_loss','val_stimu_loss','val_phosrep_loss','val_total_loss']}, epoch * len(trainloader) + i)
                    tb_writer.add_scalars(f'{model_name}/Loss/training', {key: stats[key][-1] for key in ['tr_recon_loss','tr_stimu_loss','tr_phosrep_loss','tr_total_loss']}, epoch * len(trainloader) + i)

                    sample = np.random.randint(0,sample_img.shape[0],5)
                    fig = utils.full_fig(sample_img[sample],sample_phos[sample],sample_recon[sample])
                    # fig.show()
                    tb_writer.add_figure(f'{model_name}/predictions, phosphenes and reconstruction',fig,epoch * len(trainloader) + i)  

                    
                    # 5. Save model (if best)
                    if  np.argmin(stats['val_total_loss'])+1==len(stats['val_total_loss']):
                        savepath = os.path.join(savedir,model_name + '_best_encoder.pth' )#'_e%d_encoder.pth' %(epoch))#,i))
                        logger('Saving to ' + savepath + '...')
                        torch.save(encoder.state_dict(), savepath)

                        savepath = os.path.join(savedir,model_name + '_best_decoder.pth' )#'_e%d_decoder.pth' %(epoch))#,i))
                        logger('Saving to ' + savepath + '...')
                        torch.save(decoder.state_dict(), savepath)
                        
                        for tag,img in zip(['orig','phos','recon'],[sample_img[sample],sample_phos[sample],sample_recon[sample]]):
                            savepath = os.path.join(savedir,model_name + '_'+tag+'_imgs.npy' )
                            img = img.detach().cpu().numpy()
                            with open(savepath, 'wb') as f:
                                np.save(f, img)
                        n_not_improved = 0
                    else:
                        n_not_improved = n_not_improved + 1
                        logger('not improved for %5d iterations' % n_not_improved) 
                        if n_not_improved>converg_crit:
                            break

                    # 5. Prepare for next iteration
                    encoder.train()
                    decoder.train()            

                except StopIteration:
                    pass
        if n_not_improved>converg_crit:
            break
    logger('Finished Training')
    tb_writer.close()
    return {'encoder': encoder, 'decoder':decoder}, loss_function.stats



if __name__ == '__main__':
    import pandas as pd
    
    args = utils.get_args()
    cfg = pd.Series(vars(args))
    print(cfg)
    models, dataset, optimization, train_settings = initialize_components(cfg)
    writer = SummaryWriter()
    writer.add_text("Config", cfg.to_string())
    writer.add_text("Model", 'old commit, new thresholding added')
    train(models, dataset, optimization, train_settings, writer)

