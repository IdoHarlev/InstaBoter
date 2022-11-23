import pandas as pd
from PIL import Image
import glob
import os
import csv
import shutil
from pathlib import Path
from skimage import io
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


rootdir ='/Users/ido/Desktop/CoachAI/InstaBoter/idoharlev'
#rootdir = '/Users/ido/Desktop/CoachAI/Emitions/test'
#rootdir = '/Users/ido/Desktop/CoachAI/InstaBoter/exploringwithvacations'



class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None): #, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = io.imread(img_path)
        label = torch.tensor(self.img_labels.iloc[idx , 1])
        if self.transform:
            image = self.transform(image)
        return(image,label)




class LitModel(pl.LightningModule):
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 1)
        #self.relu1 = nn.ReLU()
    '''

    def __init__(
            self, lr: float = 1e-3, num_workers: int = 4, batch_size: int = 10,
    ):
        super().__init__()
        self.lr = lr
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.conv1 = conv_block(3, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)

        self.ln1 = nn.Linear(64 * 26 * 26, 16)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout2d(0.5)
        self.ln2 = nn.Linear(16, 5)

        self.ln4 = nn.Linear(5, 10)
        self.ln5 = nn.Linear(10, 10)
        self.ln6 = nn.Linear(10, 5)
        self.ln7 = nn.Linear(10, 1)

    def forward(self, img):
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        x = img.reshape(img.shape[0], -1)
        '''
        img = self.ln1(img)
        img = self.relu(img)
        img = self.batchnorm(img)
        img = self.dropout(img)
        img = self.ln2(img)
        img = self.relu(img)
    

        x = torch.cat((img), dim=1)
        x = self.relu(x)

        x = img.reshape(img.shape[0], -1)
        '''
        return x #self.ln7(x)



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr))

    def training_step(self, batch, batch_idx):
        image, y = batch

        criterion = torch.nn.L1Loss()
        y_pred = torch.flatten(self(image))
        y_pred = y_pred.double()

        loss = criterion(y_pred, y)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

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

def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
    )

    return block



#Choose the profile root you want to explore:

#Get all pictures into folders by likes and one combined folder called "pictures"
get_jpg_from_main_dir(rootdir)
train_trans = transforms.Compose([transforms.ToTensor(),
    transforms.Resize((128,128))

    ])

#load the data into Torch
root = rootdir + '/pictures'
batch_size = 10
DB = CustomImageDataset(annotations_file='Instadata.csv', img_dir= root, transform= train_trans)
Traindata = DataLoader(dataset = DB ,shuffle= True)




model = LitModel()

logger = TensorBoardLogger("lightning_logs", name="multi_input")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=5000, patience=7, verbose=False, mode="min")



trainer = Trainer(accelerator="cpu", devices=1, logger = logger)

trainer.fit(model, Traindata)#mnist_train,mnist_val)

