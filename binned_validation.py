import pandas as pd
import numpy as np
import torch

from training_image import initialize_components
from utils import get_args, pred_to_intensities
from simulator.utils import load_params

args = get_args()
cfg = pd.Series(vars(args))
print(cfg)
models, dataset, optimization, train_settings = initialize_components(cfg)
params = load_params('simulator/config/params.yaml')
intensities_array = torch.linspace(0,params['encoding']['max_stim'],params['encoding']['n_config']+1,device=cfg.device)
input = torch.from_numpy(np.load('out/demo/exp1_chars_orig_imgs.npy')).to(args.device).float()
models['encoder'].load_state_dict(torch.load('out/demo/exp1_chars_best_encoder.pth'))
models['decoder'].load_state_dict(torch.load('out/demo/exp1_chars_best_decoder.pth'))
models['encoder'].eval()
models['decoder'].eval()
stimulation = models['encoder'](input)
stimulation = pred_to_intensities(intensities_array, stimulation)
phosphenes = models['simulator'](stimulation)
reconstruction = models['decoder'](phosphenes)

input = input.cpu().detach().numpy()
phosphenes = phosphenes.cpu().detach().numpy()
out = reconstruction.cpu().detach().numpy()

np.save('out/demo/binned_exp1_chars_orig_imgs', input)
np.save('out/demo/binned_exp1_chars_phos_imgs', phosphenes)
np.save('out/demo/binned_exp1_chars_recon_imgs', out)