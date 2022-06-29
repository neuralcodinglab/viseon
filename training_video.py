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
from simulator.utils import load_params
from simulator.init import init_probabilistically

from loss_functions import ImageLoss, ZhaoLoss

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader 

from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt

def init_models(cfg):
    models = dict()

    if cfg.model_type=='image':
        models['encoder'] = model.E2E_Encoder(in_channels=cfg.input_channels,
                                        binary_stimulation=cfg.binary_stimulation).to(cfg.device)
        models['decoder'] = model.E2E_Decoder(out_channels=cfg.reconstruction_channels,
                                        out_activation=cfg.out_activation).to(cfg.device)
    elif cfg.model_type == 'recon_pred':
        models['encoder'] = model.ZhaoEncoder(in_channels=cfg.input_channels).to(cfg.device)
        models['recon_decoder'] = model.ZhaoDecoder(out_channels=cfg.reconstruction_channels).to(cfg.device)
        models['pred_decoder'] = model.ZhaoDecoder(out_channels=cfg.reconstruction_channels).to(cfg.device)
    else:
        raise ValueError('invalid model type')

    return models

def init_loss(cfg):
    if cfg.model_type=='image':
        loss = ImageLoss(recon_loss_type=cfg.reconstruction_loss,
                                                recon_loss_param=cfg.reconstruction_loss_param,
                                                stimu_loss_type=cfg.sparsity_loss,
                                                phosrep_loss_type=cfg.representation_loss,
                                                phosrep_loss_param=cfg.representation_loss_param,
                                                kappa=cfg.kappa,
                                                device=cfg.device)
    elif cfg.model_type == 'recon_pred':
        loss = ZhaoLoss(kappa=cfg.kappa)
    
    else:
        raise ValueError('invalid model type')
    return loss

def initialize_components(cfg):
    """This function returns the required model, dataset and optimization components to initialize training.
    input: <cfg> training configuration (pandas series, or dataframe row)
    returns: dictionaries with the required model components: <models>, <datasets>,<optimization>, <train_settings>
    """

    # Random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Models
    models = init_models(cfg)
    # models = dict()
    # models['encoder'] = model.ZhaoEncoder(in_channels=cfg.input_channels).to(cfg.device)
    # models['recon_decoder'] = model.ZhaoDecoder(out_channels=cfg.reconstruction_channels).to(cfg.device)
    # models['pred_decoder'] = model.ZhaoDecoder(out_channels=cfg.reconstruction_channels).to(cfg.device)

    # Optimization
    optimization = dict()
    if cfg.optimizer == 'adam':
        for key in models:
            optimization[key] = torch.optim.Adam(models[key].parameters(),lr=cfg.learning_rate)
        # optimization['encoder'] = torch.optim.Adam(models['encoder'].parameters(),lr=cfg.learning_rate)
        # optimization['recon_decoder'] = torch.optim.Adam(models['recon_decoder'].parameters(),lr=cfg.learning_rate)
        # optimization['pred_decoder'] = torch.optim.Adam(models['pred_decoder'].parameters(),lr=cfg.learning_rate)
    elif cfg.optimizer == 'sgd':
        for key in models:
            optimization[key] = torch.optim.SGD(models[key].parameters(),lr=cfg.learning_rate)
        # optimization['encoder'] = torch.optim.SGD(models['encoder'].parameters(),lr=cfg.learning_rate)
        # optimization['recon_decoder'] = torch.optim.SGD(models['recon_decoder'].parameters(),lr=cfg.learning_rate)
        # optimization['pred_decoder'] = torch.optim.SGD(models['pred_decoder'].parameters(),lr=cfg.learning_rate)
    optimization['lossfunc'] = init_loss(cfg)  

    use_cuda = False if cfg.device=='cpu' else True
    if cfg.simulation_type == 'realistic':
        # pMap,sigma_0, activation_mask, threshold, thresh_slope, args = init_a_bit_less(use_cuda=use_cuda)

        # models['simulator'] = model.E2E_RealisticPhospheneSimulator(pMap,sigma_0, activation_mask, threshold, thresh_slope, args, device=cfg.device).to(cfg.device)
        params = load_params('simulator/config/params.yaml')
        r, phi = init_probabilistically(params,n_phosphenes=500)
        models['simulator'] = model.E2E_RealisticPhospheneSimulator(params, r, phi).to(cfg.device)
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
    if cfg.dataset == 'mnist':
        directory = './datasets/BouncingMNIST/'
        mode = 'recon' if cfg.model_type=='image' else 'recon_pred' 
        trainset = local_datasets.Bouncing_MNIST(device=cfg.device, directory=directory, mode=mode, n_frames=cfg.sequence_size)
        valset = local_datasets.Bouncing_MNIST(device=cfg.device, directory=directory, mode=mode, n_frames=cfg.sequence_size, validation = True) 
    else:
        raise ValueError

    if cfg.model_type=='image':
        print(f"setting batch size to 1, using frames as batch dimension (n_frames={cfg.sequence_size})")
        cfg.batch_size = 1

    dataset['trainloader'] = DataLoader(trainset,batch_size=int(cfg.batch_size),shuffle=True)
    dataset['valloader'] = DataLoader(valset,batch_size=int(cfg.batch_size),shuffle=False)                                 
    
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
    
def train_image_model(models, dataset, optimization, train_settings, tb_writer):
    
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
            image = data #TODO: used to be image, label
            label = None
            image = torch.transpose(image,0,2) #data is 5D, set time to batch dimension for 4d
            image = torch.squeeze(image,dim=2)
            # print(f"1 batch image shape: {image.shape}")
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
                    for i, data in enumerate(tqdm(valloader, leave=False, position=0, desc='Validation'), 0):
                        image = data #next(iter(valloader)) #TODO: handle labels
                        label = None
                        # print(label)

                        image = torch.transpose(image,0,2) #data is 5D, set time to batch dimension for 4d
                        image = torch.squeeze(image,dim=2)

                            
                        encoder.eval()
                        decoder.eval()

                        with torch.no_grad():

                            # 1. Forward pass
                            stimulation = encoder(image)
                            phosphenes  = simulator(stimulation)
                            reconstruction = decoder(phosphenes)   

                            if i==sample_iter: #save for plotting
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
                    fig.show()
                    tb_writer.add_figure(f'{model_name}/predictions, phosphenes and reconstruction',fig,epoch * len(trainloader) + i)  

                    
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

                except StopIteration:
                    pass
        if n_not_improved>converg_crit:
            break
    logger('Finished Training')
    tb_writer.close()
    return {'encoder': encoder, 'decoder':decoder}, loss_function.stats

def train_recon_pred_model(models, dataset, optimization, train_settings, tb_writer):
    
    ## A. Unpack parameters
   
    # Models
    encoder         = models['encoder']
    recon_decoder   = models['recon_decoder']
    pred_decoder    = models['pred_decoder']
    simulator = models['simulator']
    
    # Dataset
    trainloader = dataset['trainloader']
    valloader   = dataset['valloader']
    
    # Optimization
    encoder_optim = optimization['encoder']
    recon_decoder_optim = optimization['recon_decoder']
    pred_decoder_optim = optimization['pred_decoder']
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

        for i, data in enumerate(tqdm(trainloader), 0):
            input_frames,future_frames = data
            # plt.imshow(input_frames[0,0,0,:,:])
            # plt.colorbar()
            # plt.show()
            # TRAINING
            encoder.train()
            recon_decoder.train()
            pred_decoder.train()
            encoder.zero_grad()
            recon_decoder.zero_grad()
            pred_decoder.zero_grad()

            # 1. Forward pass
            # print(f"input: {input_frames.shape}")
            stimulation = encoder(input_frames)
            # print(f"stimulation: {stimulation.shape}")
            # proxy_stim = 30*torch.ones_like(stimulation)
            # phosphenes  = simulator(stimulation)
            phosphenes=stimulation
            # print(f"phosphenes: {phosphenes.shape}")
            reconstruction = recon_decoder(phosphenes)
            # print(f"reconstruction: {reconstruction.shape}")
            prediction = pred_decoder(phosphenes)
            # print(f"prediction: {prediction.shape}")
            # print(f"passed sample through models {count_samples} times")
            count_samples+=1
            # 2. Calculate loss
            loss = loss_function(input_frames = input_frames,
                                 future_frames = future_frames,
                                 phosphenes=phosphenes,
                                 reconstruction=reconstruction,
                                 prediction=prediction)
            # print("calculated loss")
            
            # 3. Backpropagation
            loss.backward()
            encoder_optim.step()
            recon_decoder_optim.step()
            pred_decoder_optim.step()
            # print(f"optimizer step {count_backwards}")
            # count_backwards+=1

            # running_loss += loss.item()

            # VALIDATION
            if i==len(trainloader) or i % log_interval == 0:
                # print(f"{count_val} times in validation loop")
                # tb_writer.add_scalar('Loss/train', running_loss/log_interval, epoch * len(trainloader) + i)
                # running_loss = 0.0
                tb_writer.add_histogram(f'{model_name}/stimulation',stimulation,epoch * len(trainloader) + i)
                utils.log_gradients_in_model(encoder, f'{model_name}/encoder', tb_writer, epoch * len(trainloader) + i)
                utils.log_gradients_in_model(recon_decoder, f'{model_name}/recon_decoder', tb_writer, epoch * len(trainloader) + i)
                utils.log_gradients_in_model(pred_decoder, f'{model_name}/pred_decoder', tb_writer, epoch * len(trainloader) + i)
                # print(stimulation)
                count_val+=1
                try:
                    input_frames,future_frames = next(iter(valloader))
                    # print(label)

                    encoder.eval()
                    recon_decoder.eval()
                    pred_decoder.eval()

                    with torch.no_grad():

                        # 1. Forward pass
                        stimulation = encoder(input_frames)
                        # phosphenes  = simulator(stimulation)
                        phosphenes = stimulation
                        reconstruction = recon_decoder(phosphenes)
                        prediction = pred_decoder(phosphenes)

                        # 2. Loss
                        stats = loss_function(input_frames = input_frames,
                                            future_frames = future_frames,
                                            phosphenes=phosphenes,
                                            reconstruction=reconstruction,
                                            prediction=prediction,
                                            validation=True)            
  
                    tb_writer.add_scalars(f'{model_name}/Loss/validation', {key: stats[key][-1] for key in ['val_recon_loss','val_pred_loss','val_total_loss']}, epoch * len(trainloader) + i) #,'val_phosrep_loss'
                    tb_writer.add_scalars(f'{model_name}/Loss/training', {key: stats[key][-1] for key in ['tr_recon_loss','tr_pred_loss','tr_total_loss']}, epoch * len(trainloader) + i) #,'tr_phosrep_loss'
                    # sample = np.random.randint(0,input_frames.shape[0],5)
                    sample = np.random.randint(0,input_frames.shape[0],1)
                    fig = utils.full_fig(input_frames[sample,:,:5],reconstruction[sample,:,:5],future_frames[sample,:,:5],prediction[sample,:,:5]) #phosphenes[sample]
                    fig.show()
                    tb_writer.add_figure(f'{model_name}/predictions, phosphenes,reconstruction',fig,epoch * len(trainloader) + i)  

                    
                    # 5. Save model (if best)
                    if  np.argmin(stats['val_total_loss'])+1==len(stats['val_total_loss']):
                        savepath = os.path.join(savedir,model_name + '_best_encoder.pth' )#'_e%d_encoder.pth' %(epoch))#,i))
                        logger('Saving to ' + savepath + '...')
                        torch.save(encoder.state_dict(), savepath)

                        savepath = os.path.join(savedir,model_name + '_best_recon_decoder.pth' )#'_e%d_decoder.pth' %(epoch))#,i))
                        logger('Saving to ' + savepath + '...')
                        torch.save(recon_decoder.state_dict(), savepath)

                        savepath = os.path.join(savedir,model_name + '_best_pred_decoder.pth' )#'_e%d_decoder.pth' %(epoch))#,i))
                        logger('Saving to ' + savepath + '...')
                        torch.save(pred_decoder.state_dict(), savepath)
                        
                        n_not_improved = 0
                    else:
                        n_not_improved = n_not_improved + 1
                        logger('not improved for %5d iterations' % n_not_improved) 
                        if n_not_improved>converg_crit:
                            break

                    # 5. Prepare for next iteration
                    encoder.train()
                    recon_decoder.train()
                    pred_decoder.train()

                except StopIteration:
                    pass
        if n_not_improved>converg_crit:
            break

    logger('Finished Training')

    tb_writer.close()
    
    return {'encoder': encoder, 'recon_decoder':recon_decoder, 'pred_decoder':pred_decoder}, loss_function.stats



if __name__ == '__main__':
    import pandas as pd
    
    args = utils.get_args()
    cfg = pd.Series(vars(args))
    print(cfg)
    models, dataset, optimization, train_settings = initialize_components(cfg)
    writer = SummaryWriter()
    if cfg.model_type=='image':
        train_image_model(models, dataset, optimization, train_settings, writer)
    elif cfg.model_type=='recon_pred':
        train_recon_pred_model(models, dataset, optimization, train_settings, writer)
    else:
        raise ValueError('invalid model type')

