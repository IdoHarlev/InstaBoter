import pandas as pd
import numpy as np
import torch
from PIL import Image
import glob
import os
import csv
from torchvision.io import read_image
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import shutil
from pathlib import Path

rootdir ='/Users/ido/Desktop/CoachAI/InstaBoter/idoharlev'
rootdir1 = '/Users/ido/Desktop/CoachAI/Emitions/test'
rootdir2 = '/Users/ido/Desktop/CoachAI/InstaBoter/exploringwithvacations'



class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None): #, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label =  self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #    label = self.target_transform(label)
        return image, label





def get_jpg_from_main_dir(mainroot):
     with open('Instadata.csv', 'w', encoding='UTF8') as f:
          writer = csv.writer(f)
          writer.writerow(['file_name','Num_Likes'])
     subdirs = [x[1] for x in os.walk(mainroot)]
     #print(subdirs[0])

     directory = "pictures"

     path = os.path.join(mainroot, directory)
     if not os.path.isdir(path):
         os.makedirs(path)
     with open('Instadata.csv', 'w', encoding='UTF8') as f:
          writer = csv.writer(f)
          for i,k in enumerate(subdirs[0]):
               root = mainroot + '/' + str(k) + '/*.jpg'
               jpgFilenamesList = glob.glob(root)
               for j in jpgFilenamesList:
                   if k == 'pictures':
                       continue
                   else:

                       writer.writerow([Path(j).name, k])
                       # print(j,k)
                       image = Image.open(j)
                       # image.show()
                   try:
                        shutil.copy(j, path)
                   except shutil.SameFileError:
                       pass





#Choose the profile root you want to explore:

#Get all pictures into folders by likes and one combined folder called "pictures"
get_jpg_from_main_dir(rootdir)
train_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,640)),
    transforms.ToTensor()
    ])

#load the data into Torch
root = rootdir + '/pictures'
DB = CustomImageDataset(annotations_file='Instadata.csv', img_dir= root, transform= train_trans)
Traindata = DataLoader(dataset = DB ,shuffle= True)






