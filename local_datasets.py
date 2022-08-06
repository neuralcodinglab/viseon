import numpy as np
import os
import pickle
from glob import glob
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.datasets as ds
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageFilter 
import cv2 as cv
import string
import utils
from tqdm import tqdm

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    x = torch.arange(h)
    Y, X = torch.meshgrid(x,x)
    dist_from_center = torch.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

class Bouncing_MNIST(Dataset):

    def __init__(self, directory='./datasets/BouncingMNIST',
                 device = torch.device('cuda:0'),
                 mode = 'recon',
                 imsize=128,
                 n_frames=6,
                 validation=False):
        super().__init__()

        self.device = device
        self.mode = mode
        self.imsize = imsize
        self.n_frames = n_frames
        # self.validation = validation

        full_set = np.load(directory+'mnist_test_seq.npy').transpose(1, 0, 2, 3)
        # full_set = full_set[:100] #TODO: remove
        
        if self.mode=='recon':
            if n_frames<full_set.shape[1]:
                divisor = int(full_set.shape[1]/n_frames)
                full_set = full_set[:,:n_frames*divisor,:,:]
                full_set = full_set.reshape((-1,n_frames,full_set.shape[2],full_set.shape[3]))
        # print(full_set.shape)
        np.random.shuffle(full_set)
        split_int = int(0.1*full_set.shape[0])
        if validation:
            self.input = torch.from_numpy(full_set[:split_int])
        else:
            self.input = torch.from_numpy(full_set[split_int:])

        self.input = self.input.unsqueeze(dim=1)
        print(f"input shape: {self.input.shape}")


    def __len__(self):
        return len(self.input)

    def __getitem__(self, i):
        if self.mode == 'recon':
            frames = T.Resize(128)(self.input[i]/255.)
            return frames.detach().to(self.device)
        elif self.mode == 'recon_pred':
            input_frames = T.Resize(128)(self.input[i,:,:self.n_frames]/255.)#.to(self.device)
            future_frames = T.Resize(128)(self.input[i,:,self.n_frames:self.n_frames*2]/255.)#.to(self.device)
            return input_frames.detach().to(self.device), future_frames.detach().to(self.device)
            
class ADE_Dataset(Dataset):
    
    def __init__(self, directory='../_Datasets/ADE20K/',
                 device=torch.device('cuda:0'),
                 imsize = 128,
                 grayscale = True,
                 normalize = True,
                 contour_labels = True,
                 validation=False,
                 load_preprocessed=False,
                 circular_mask=True):
        
        self.contour_labels = contour_labels
        self.normalize = normalize
        self.grayscale = grayscale
        self.device = device
        self.circular_mask = circular_mask
    
        contour = lambda im: im.filter(ImageFilter.FIND_EDGES).point(lambda p: p > 1 and 255) if self.contour_labels else im
        # to_grayscale = lambda im: im.convert('L') if self.grayscale else im

        # Image and target tranformations (square crop and resize)
        self.img_transform = T.Compose([T.Lambda(lambda img:F.center_crop(img, min(img.size))),
                                        T.Resize(imsize),
                                        T.ToTensor()
                                        ])
        self.trg_transform = T.Compose([T.Lambda(lambda img:F.center_crop(img, min(img.size))),
                                        T.Resize(imsize,interpolation=T.InterpolationMode.NEAREST), # Nearest Neighbour
                                        T.Lambda(contour),
                                        T.ToTensor()
                                        ])
        
        # Normalize
        self.normalizer = T.Normalize(mean = [0.485, 0.456, 0.406],
                                      std = [0.229, 0.224, 0.225])        
        
        # RGB converter
        weights=[.3,.59,.11]
        self.to_grayscale = lambda image:torch.sum(torch.stack([weights[c]*image[c,:,:] for c in range(3)],dim=0),
                                                   dim=0,
                                                   keepdim=True)

        self.inputs = []
        self.targets = []
        
        val_path = '_val/' if validation else '_train/'
        path = directory+'/images/processed'+val_path
        if load_preprocessed:
            self.load(path)
            print('----Loaded preprocessed data----')
            print(f'input length: {len(self.inputs)} samples')
        else:
            # Collect files 
            img_files, seg_files = [],[]
            print('----Listing training images----')
            for path, subdirs, files in tqdm(os.walk(os.path.join(directory,'images','training'))):
            # for path, subdirs, files in os.walk(os.path.join(directory,'training')):
                img_files+= glob(os.path.join(path,'*.jpg'))
                seg_files+= glob(os.path.join(path,'*seg.png'))
                val_img_files, val_seg_files, = [],[]
            print('----Listing validation images----')
            for path, subdirs, files in tqdm(os.walk(os.path.join(directory,'images','validation'))):
            # for path, subdirs, files in os.walk(os.path.join(directory,'validation')):
                val_img_files+= glob(os.path.join(path,'*.jpg'))
                val_seg_files+= glob(os.path.join(path,'*seg.png'))
            for l in [img_files,seg_files,val_img_files,val_seg_files]:
                l.sort()

            print('Finished listing files')
            # Image and target files
            if validation:
                self.input_files = val_img_files
                self.target_files = val_seg_files
            else:
                self.input_files = img_files
                self.target_files = seg_files

            print('----Preprocessing ADE20K input----')
            for image, target in tqdm(zip(self.input_files, self.target_files),total=len(self.input_files)):
                im = Image.open(image).convert('RGB')
                t = Image.open(target).convert('L')

                # Crop, resize & transform
                x = self.img_transform(im)
                t = self.trg_transform(t)
                                            
                # Additional tranforms:
                if self.normalize:
                    x = self.normalizer(x)
                if self.grayscale:
                    x = self.to_grayscale(x)

                self.inputs += [x]
                self.targets += [t]
            print('----Finished preprocessing----')
            self.save(path)

    def save(self,path):
        with open(path+'_inputs.pkl','wb') as f:
            pickle.dump(self.inputs,f)
        with open(path+'_targets.pkl','wb') as f:
            pickle.dump(self.targets,f)

    def load(self,path):
        with open(path+'_inputs.pkl','rb') as f:
            self.inputs = pickle.load(f)
        with open(path+'_targets.pkl','rb') as f:
            self.targets = pickle.load(f)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        
        x = self.inputs[i]
        t = self.targets[i]

        if self.circular_mask:
            mask = create_circular_mask(x.shape[1],x.shape[2]).view(1,x.shape[1],x.shape[2])
            x = x*mask
            t = t*mask
    
        return x.detach().to(self.device),t.detach().to(self.device)
    
    
class Character_Dataset(Dataset):
    """ Pytorch dataset containing images of single (synthetic) characters.
    __getitem__ returns an image containing one of 26 ascci lowercase characters, 
    typed in one of 47 fonts(default: 38 train, 9 validation) and the corresponding
    alphabetic index as label.
    """
    def __init__(self,directory = './datasets/Characters/',
                 device=torch.device('cuda:0'),
                 imsize = (128,128),
                 train_val_split = 0.8,
                 validation=False,
                 word_scale=.8,
                 invert = True): 
        
        self.imsize = imsize
        self.tensormaker = T.ToTensor()
        self.device = device
        self.validation = validation
        self.word_scale = word_scale
        self.invert = invert
        
        characters = string.ascii_lowercase
        fonts = glob(os.path.join(directory,'Fonts/*.ttf'))
        
        self.split = round(len(fonts)*train_val_split)
        train_data, val_data = [],[]
        for c in characters:
            for f in fonts[:self.split]:
                train_data.append((f,c))
            for f in fonts[self.split:]:
                val_data.append((f,c))
        self.data = val_data if validation else train_data
        self.classes = characters
        self.lookupletter = {letter: torch.tensor(index) for index, letter in enumerate(characters)}
        self.padding_correction = 6 #By default, PILs ImageDraw function uses excessive padding                                                          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        # Load font and character
        f,c = self.data[i]
        
        # Get label (alphabetic index of character)
        lbl  = self.lookupletter[c]
        
        # Scale character to image
        fontsize = 1
        font = ImageFont.truetype(f,fontsize)
        while max(font.getsize(c))/min(self.imsize) <= self.word_scale:
            fontsize += 1
            font = ImageFont.truetype(f,fontsize)
        fontsize -=1 

        # Calculate left-over space 
        font = ImageFont.truetype(f,fontsize)
        textsize = font.getsize(c)
        free_space = np.subtract(self.imsize,textsize)
        free_space += self.padding_correction

        # Draw character at random position
        img = Image.fromarray(255*np.ones(self.imsize).astype('uint8'))
        draw = ImageDraw.Draw(img)
        location = np.random.rand(2)*(free_space)
        location[1]-= self.padding_correction
        draw.text(location,c,(0,),font=font)       
        img = self.tensormaker(img)
        
        
        if self.invert:
            img = 1-img
        return img.to(self.device), lbl.to(self.device)

