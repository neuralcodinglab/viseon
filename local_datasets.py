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

class ADE_Dataset(Dataset):
    
    def __init__(self, directory='../_Datasets/ADE20K/',
                 device=torch.device('cuda:0'),
                 imsize = 128,
                 grayscale = True,
                 normalize = True,
                 contour_labels = True,
                 validation=False,
                 load_preprocessed=False):
        
        self.contour_labels = contour_labels
        self.normalize = normalize
        self.grayscale = grayscale
        self.device = device
        # print('before first list comprehension')

        

        # img_files, seg_files = [],[]
        # for path, subdirs, files in os.walk(os.path.join(directory,'images','training')):
        #     img_files+= [files for files in glob(os.path.join(path,'*.jpg'))]
        #     seg_files+= [files for files in glob(os.path.join(path,'*seg.png'))]
        # val_img_files, val_seg_files, = [],[]
        # for path, subdirs, files in os.walk(os.path.join(directory,'images','validation')):
        #     val_img_files+= [files for files in glob(os.path.join(path,'*.jpg'))] 
        #     val_seg_files+= [files for files in glob(os.path.join(path,'*seg.png'))]
        # for l in [img_files,seg_files,val_img_files,val_seg_files]:
        #     l.sort()   
        
    
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
        
        # Transform sem. seg. to contour labels
        # to_grayscale = self.to_grayscale #lambda image : image.mean(axis=0,keepdims=True)
        # to_cv2_list  = lambda image_tensor : [np.squeeze((255*img.cpu().numpy())).astype('uint8') for img in image_tensor]
        # canny_edge   = lambda image_list : [cv.Canny(img,10,10)/255 for img in image_list]
        # to_tensor    = lambda image_list : torch.tensor(np.array(image_list), dtype=torch.long)
        # squeeze_img  = lambda image_tensor : torch.squeeze(image_tensor)
        # self.contour = T.Compose([T.Lambda(to_grayscale),
        #                           T.Lambda(to_cv2_list),
        #                           T.Lambda(canny_edge),
        #                           T.Lambda(to_tensor),
        #                           T.Lambda(squeeze_img)])


        # contour_fun = lambda im: im.filter(ImageFilter.FIND_EDGES).point(lambda p: p > 1 and 255)
        # gray_resize = lambda im: im.convert("L").resize((128,128))

        
        # center_crop = lambda img:F.center_crop(img, min(img.size))
        # self.trg_transform = T.Compose([T.Lambda(center_crop),
        #                                 T.Resize(imsize,interpolation=Image.NEAREST),# Nearest Neighbour
        #                                 T.Lambda(contour),
        #                                 T.ToTensor()])

        # self.contour = T.Compose([T.Lambda(contour_fun), T.Lambda(gray_resize)])

        self.inputs = []
        self.targets = []
        
        val_path = '_val' if validation else '_train'
        path = directory+'processed'+val_path
        if load_preprocessed:
            self.load(path)
            print('----Loaded preprocessed data----')
            print(f'input length: {len(self.inputs)} samples')
            # self.inputs = self.inputs[:10000]
            # self.targets = self.targets[:10000]
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
        # # Load Image, Label
        # x = Image.open(self.input_files[i]).convert('RGB')
        # t = Image.open(self.target_files[i])
        

        
        # # Crop and resize
        # x = self.img_transform(x)
        # t = self.trg_transform(t)
                                      
        # # Additional tranforms:
        # if self.normalize:
        #     x = self.normalizer(x)
        # if self.contour_labels:
        #     t = self.contour(t)
        # if self.grayscale:
        #     x = self.to_grayscale(x)
        

            
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

