import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import random_split
from torchvision.transforms import transforms
import Reader
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar


class LitModel(pl.LightningModule):
    def __init__(self, image_shape = (3,28,28), hidden_units = (32,16)):
        super().__init__()
        self.model = nn.Linear(3,128,128)


    def forward(self, x):
        X = x.view(x.size(0), -1)
        print(x.shape)
        X = self.model(x)

        return x



    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    '''def validation_step (self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)'''


root = Reader.rootdir
Reader.get_jpg_from_main_dir(root)
train_trans = transforms.Compose([transforms.ToTensor(),
    #transforms.ToPILImage(),
    transforms.Resize((128,128))

    ])

#load the data into Torch
root1 = root + '/pictures'
DB = Reader.CustomImageDataset(annotations_file='Instadata.csv', img_dir= root1, transform= train_trans)
Traindata = DataLoader(dataset = DB ,shuffle= True)
#mnist_train, mnist_val = random_split(Traindata, [50, 10])

'''\
model = LitModel()

trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=3,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
)
trainer.fit(model, Traindata)#mnist_train,mnist_val)
'''

net = LitModel()
x = torch.randn(1, 3, 28, 28)
out = net(x)
print(out.shape)
