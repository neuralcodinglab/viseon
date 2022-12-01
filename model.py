import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

def get_e2e_autoencoder(cfg):

    # initialize encoder and decoder
    if cfg['output_steps'] is not None:
        encoder = torch.nn.Sequential(E2E_Encoder(in_channels=cfg['in_channels'],
                                                        n_electrodes=cfg['n_electrodes'],
                                                        out_scaling=1.,
                                                        out_activation='sigmoid'),
                                      SafetyLayer(n_steps=10,
                                                        order=2,
                                                        out_scaling=cfg['output_scaling'])).to(cfg['device'])
    else:
        encoder = E2E_Encoder(in_channels=1,
                                    n_electrodes=cfg['n_electrodes'],
                                    out_scaling=cfg['output_scaling'],
                                    out_activation='relu').to(cfg['device'])

    decoder = E2E_Decoder(out_channels=cfg['out_channels'],
                          out_activation=cfg['out_activation']).to(cfg['device'])

    return encoder, decoder

def get_Zhao_autoencoder(cfg):
    encoder = ZhaoEncoder(in_channels=cfg['in_channels'], n_electrodes=cfg['n_electrodes']).to(cfg['device'])
    decoder = ZhaoDecoder(out_channels=cfg['out_channels'], out_activation=cfg['out_activation']).to(cfg['device'])

    return encoder, decoder

def convlayer(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    layer = [
        nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer


def convlayer3d(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    layer = [
        nn.Conv3d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer 

def deconvlayer3d(n_input, n_output, k_size=2, stride=2, padding=0, dilation=1, resample_out=None):
    layer = [
        nn.ConvTranspose3d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm3d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.resample_out = resample_out

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if self.resample_out:
            out = self.resample_out(out)
        return out
    
  
class SafetyLayer(torch.nn.Module):
    def __init__(self, n_steps=5, order=1, out_scaling=120e-6):
        super(SafetyLayer, self).__init__()
        self.n_steps = n_steps
        self.order = order
        self.output_scaling = out_scaling

    def stairs(self, x):
        """Assumes input x in range [0,1]. Returns quantized output over range [0,1] with n quantization levels"""
        return torch.round((self.n_steps-1)*x)/(self.n_steps-1)

    def softstairs(self, x):
        """Assumes input x in range [0,1]. Returns sin(x) + x (soft staircase), scaled to range [0,1].
        param n: number of phases (soft quantization levels)
        param order: number of recursion levels (determining the steepnes of the soft quantization)"""

        return (torch.sin(((self.n_steps - 1) * x - 0.5) * 2 * math.pi) +
                         (self.n_steps - 1) * x * 2 * math.pi) / ((self.n_steps - 1) * 2 * math.pi)
    
    def forward(self, x):
        out = self.softstairs(x) + self.stairs(x).detach() - self.softstairs(x).detach()
        return (out * self.output_scaling).clamp(1e-32,None)


class VGGFeatureExtractor():
    def __init__(self,layer_names=['1','3','6','8'], layer_depth=9 ,device='cpu'):
        
        # Load the VGG16 model
        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = torch.nn.Sequential(*[*model.features][:layer_depth]).to(device)
        
        # Register a forward hook for each layer of interest
        self.layers = {name: layer for name, layer in self.feature_extractor.named_children() if name in layer_names}
        self.outputs = dict()
        for name, layer in self.layers.items():
            layer.__name__ = name
            layer.register_forward_hook(self.store_output)
            
    def store_output(self, layer, input, output):
        self.outputs[layer.__name__] = output

    def __call__(self, x):
        
        # If grayscale, convert to RGB
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        
        # Forward pass
        self.feature_extractor(x)
        activations = list(self.outputs.values())
        
        return activations
        
        

class E2E_Encoder(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu'):
        super(E2E_Encoder, self).__init__()
        self.output_scaling = out_scaling
        self.out_activation = {'tanh': nn.Tanh(), ## NOTE: simulator expects only positive stimulation values 
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.ReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        # Model
        self.model = nn.Sequential(*convlayer(in_channels,8,3,1,1),
                                   *convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   *convlayer(32,16,3,1,1),
                                   nn.Conv2d(16,1,3,1,1),
                                   nn.Flatten(),
                                   nn.Linear(1024,n_electrodes),
                                   self.out_activation)

    def forward(self, x):
        self.out = self.model(x)
        stimulation = self.out*self.output_scaling #scaling improves numerical stability
        return stimulation

class E2E_Decoder(nn.Module):
    """
    Simple non-generic phosphene decoder.
    in: (256x256) SVP representation
    out: (128x128) Reconstruction
    """
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(E2E_Decoder, self).__init__()

        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        # Model
        self.model = nn.Sequential(*convlayer(in_channels,16,3,1,1),
                                   *convlayer(16,32,3,1,1),
                                   *convlayer(32,64,3,2,1),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   *convlayer(64,32,3,1,1),
                                   nn.Conv2d(32,out_channels,3,1,1),
                                   self.out_activation)

    def forward(self, x):
        return self.model(x)

# class E2E_RealisticPhospheneSimulator(nn.Module):
#     """A realistic simulator, using stimulation vectors to form a phosphene representation
#     in: a 1024 length stimulation vector
#     out: 256x256 phosphene representation
#     """
#     def __init__(self, cfg, params, r, phi):
#         super(E2E_RealisticPhospheneSimulator, self).__init__()
#         self.simulator = GaussianSimulator(params, r, phi, batch_size=cfg.batch_size, device=cfg.device)
        
#     def forward(self, stimulation):
#         phosphenes = self.simulator(stim_amp=stimulation).clamp(0,1)
#         phosphenes = phosphenes.view(phosphenes.shape[0], 1, phosphenes.shape[1], phosphenes.shape[2])
#         return phosphenes

class ZhaoEncoder(nn.Module):
    def __init__(self, in_channels=3,n_electrodes=638, out_channels=1):
        super(ZhaoEncoder, self).__init__()

        self.model = nn.Sequential(
            *convlayer3d(in_channels,32,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(32,48,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(48,64,3,1,1),
            *convlayer3d(64,1,3,1,1),

            nn.Flatten(start_dim=3),
            nn.Linear(1024,n_electrodes),
            nn.ReLU()
        )

    def forward(self, x):
        self.out = self.model(x)
        self.out = self.out.squeeze(dim=1)
        self.out = self.out*1e-4
        return self.out

class ZhaoDecoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(ZhaoDecoder, self).__init__()
        
        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        self.model = nn.Sequential(
            *convlayer3d(in_channels,16,3,1,1),
            *convlayer3d(16,32,3,1,1),
            *convlayer3d(32,64,3,(1,2,2),1),
            *convlayer3d(64,32,3,1,1),
            nn.Conv3d(32,out_channels,3,1,1),
            self.out_activation
        )

    def forward(self, x):
        self.out = self.model(x)
        return self.out
