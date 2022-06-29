from syslog import LOG_SYSLOG
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import math
import model_video as model

class ZhaoLoss(object):
    def __init__(self, kappa=None):
        """Custom loss class for training end-to-end model for videos. Based on Zhao et al 2017: Spatio-Temporal AutoEncoder for Video Anomaly Detection. 
        Uses a reconstruction loss where the original video is reconstructed and a prediction loss for the next few frames.
        """
        
        # Reconstruction loss
        self.recon_loss = torch.nn.MSELoss()
        
        self.pred_loss = torch.nn.MSELoss()

        # self.phosrep_loss = torch.nn.MSELoss()
        
        self.kappa = kappa

        # Output statistics 
        self.stats = {
            'tr_recon_loss':[],
            'val_recon_loss': [],
            'tr_pred_loss': [],
            'val_pred_loss': [],
            # 'tr_phosrep_loss': [],
            # 'val_phosrep_loss': [],
            'tr_total_loss':[],
            'val_total_loss':[]} 
        self.running_loss = {'recon':0,'pred':0,'total':0} #,'phosrep':0
        self.n_iterations = 0
        
        
    def __call__(self,input_frames,future_frames,phosphenes,reconstruction,prediction,validation=False):   

        # print(f"phosphenes shape: {phosphenes.shape}") 
        
        # phs = T.Resize(size=(input_frames.size()[-2],input_frames.size()[-1]))(phosphenes)

        # Calculate loss
        # loss_phosrep = self.phosrep_loss(phs,input_frames)
        loss_recon = self.recon_loss(reconstruction,input_frames)
        loss_pred = self.pred_loss(prediction,future_frames)

        # if self.kappa is None:
        #     loss_total = loss_phosrep/3 + loss_recon/3 + loss_pred/3
        # else:
        #     loss_total = loss_phosrep*self.kappa[0] + loss_recon*self.kappa[1] + loss_pred*self.kappa[2]
        loss_total = 0.5*loss_recon+0.5*loss_pred
        if not validation:
            # Save running loss and return total loss
            self.running_loss['pred'] += loss_pred.item()
            self.running_loss['recon'] += loss_recon.item()
            # self.running_loss['phosrep'] += loss_phosrep.item()
            self.running_loss['total'] += loss_total.item()
            self.n_iterations += 1
            return loss_total
        else:
            self.stats['val_recon_loss'].append(loss_recon.item())
            self.stats['val_pred_loss'].append(loss_pred.item())
            # self.stats['val_phosrep_loss'].append(loss_phosrep.item())
            self.stats['val_total_loss'].append(loss_total.item())
            self.stats['tr_recon_loss'].append(self.running_loss['recon']/self.n_iterations)
            self.stats['tr_pred_loss'].append(self.running_loss['pred']/self.n_iterations)
            # self.stats['tr_phosrep_loss'].append(self.running_loss['phosrep']/self.n_iterations)  
            self.stats['tr_total_loss'].append(self.running_loss['total']/self.n_iterations)
                
            self.running_loss = {key:0 for key in self.running_loss}
            self.n_iterations = 0
            return self.stats

class ImageLoss(object):
    def __init__(self, recon_loss_type='mse',recon_loss_param=None, stimu_loss_type=None, phosrep_loss_type=None, phosrep_loss_param=None, kappa=0, device='cpu'):
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
            # loss_weights = torch.tensor([1-recon_loss_param,recon_loss_param],device=device)
            # self.recon_loss = torch.nn.CrossEntropyLoss()#weight=loss_weights)
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
        
        # Posphene representation loss
        if phosrep_loss_type=='mse':
            self.phosrep_loss = torch.nn.MSELoss()
        elif phosrep_loss_type=='ssim':
            self.phosrep_loss = ssim_loss
        elif phosrep_loss_type == 'vgg':
            self.feature_extractor = model.VGG_Feature_Extractor(layer_depth=phosrep_loss_param,device=device)
            self.phosrep_loss = lambda x,y: torch.nn.functional.mse_loss(self.feature_extractor(x),self.feature_extractor(y))
        else:
            self.phosrep_loss = None
            
        self.kappa = kappa if self.stimu_loss or self.phosrep_loss else 0

        # Output statistics 
        self.stats = {'tr_recon_loss':[],'val_recon_loss': [],'tr_total_loss':[],'val_total_loss':[]}
        # if self.stimu_loss is not None:   
        self.stats['tr_stimu_loss']= []
        self.stats['val_stimu_loss']= []
        # if self.phosrep_loss is not None:   
        self.stats['tr_phosrep_loss']= []
        self.stats['val_phosrep_loss']= [] 
        self.running_loss = {'recon':0,'stimu':0,'phosrep':0,'total':0}
        self.val_running_loss = {'recon':0,'stimu':0,'phosrep':0,'total':0}
        self.n_iterations = 0
        self.n_val_iterations = 0
        
    def get_stats(self,reset_train=False,reset_val=False):
        self.stats['val_recon_loss'].append(self.val_running_loss['recon']/self.n_val_iterations)
        self.stats['val_total_loss'].append(self.val_running_loss['total']/self.n_val_iterations)
        self.stats['tr_recon_loss'].append(self.running_loss['recon']/self.n_iterations)
        self.stats['tr_total_loss'].append(self.running_loss['total']/self.n_iterations)
        # if self.stimu_loss is not None:
        self.stats['val_stimu_loss'].append(self.val_running_loss['stimu']/self.n_val_iterations)
        self.stats['tr_stimu_loss'].append(self.running_loss['stimu']/self.n_iterations)
        # if self.phosrep_loss is not None:
        self.stats['val_phosrep_loss'].append(self.val_running_loss['phosrep']/self.n_val_iterations)
        self.stats['tr_phosrep_loss'].append(self.running_loss['phosrep']/self.n_iterations)  

        if reset_train:
            self.running_loss = {key:0 for key in self.running_loss}
            self.n_iterations = 0
        if reset_val:
            self.val_running_loss = {key:0 for key in self.val_running_loss}
            self.n_val_iterations = 0
        return self.stats

    def __call__(self,image,label,stimulation,phosphenes,reconstruction,validation=False):    
        
        # Target
        if self.target == 'image': # Flag for reconstructing input image or target label
            target = image
        elif self.target == 'label':
            target = label
        
        phs = T.Resize(size=(image.size()[-1],image.size()[-2]))(phosphenes)

        # Calculate loss
        loss_stimu = self.stimu_loss(stimulation) if self.stimu_loss else torch.tensor(0)
        loss_recon = self.recon_loss(reconstruction,target)
        loss_phosrep = self.phosrep_loss(phs,image) if self.phosrep_loss else torch.tensor(0)

        #TODO: either use stimulation loss or phosphene representation loss? Or change kappa to 3-way split 
        if self.phosrep_loss:
            loss_total = (1-self.kappa)*loss_recon + self.kappa*loss_phosrep
        else:
            loss_total = (1-self.kappa)*loss_recon + self.kappa*loss_stimu
        
        if not validation:
            # Save running loss and return total loss
            self.running_loss['stimu'] += loss_stimu.item()
            self.running_loss['recon'] += loss_recon.item()
            self.running_loss['phosrep'] += loss_phosrep.item()
            self.running_loss['total'] += loss_total.item()
            self.n_iterations += 1
            return loss_total
        else:
            self.val_running_loss['stimu'] += loss_stimu.item()
            self.val_running_loss['recon'] += loss_recon.item()
            self.val_running_loss['phosrep'] += loss_phosrep.item()
            self.val_running_loss['total'] += loss_total.item()
            self.n_val_iterations += 1
            return loss_total

class Representation_Loss(object):
    def __init__(self,loss_type='mse') -> None:
        if loss_type=='mse':
            self.phosrep_loss = torch.nn.MSELoss()
        elif loss_type=='ssim':
            self.phosrep_loss = ssim_loss
        else:
            raise ValueError

        self.stats = {'tr_loss':[],'val_loss': []}
        self.running_loss = {'total':0}
        self.n_iterations = 0

    def __call__(self,image,phosphenes,validation=False):
        phs = T.Resize(size=(image.size()[-3],image.size()[-2],image.size()[-2]))(phosphenes)

        loss_phosrep = self.phosrep_loss(phs,image)

        if not validation:
            # Save running loss and return total loss
            self.running_loss['total'] += loss_phosrep.item()
            self.n_iterations += 1
            return loss_phosrep
        else:
            self.stats['val_loss'].append(loss_phosrep.item())
            self.stats['tr_loss'].append(self.running_loss['total']/self.n_iterations)  

            self.running_loss = {key:0 for key in self.running_loss}
            self.n_iterations = 0
            return self.stats

def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = torch.Tensor(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )

    return window


def ssim_loss(
    img1, img2, window_size=11, window=None, size_average=True, full=False
):
    # print("size: ",img2.size())
    # L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2

    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None:
        real_size = min(window_size, height, width)  # window should be atleast 11x11
        window = create_window(real_size, channel=channels).to(img1.device)

    # calculating the mu parameter (locally) for both images using a gaussian filter
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability
    C1 = (0.01) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03) ** 2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean()
    else:
        ret = ssim_score.mean(1).mean(1).mean(1)

    loss = 1-(ret+1)/2
    if full:
        return loss, contrast_metric

    return loss