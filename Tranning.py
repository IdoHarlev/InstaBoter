import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import random_split
from torchvision.transforms import transforms
import Reader
import Reader as rd

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        print(loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


root = Reader.rootdir
Reader.get_jpg_from_main_dir(root)
train_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,640)),
    transforms.ToTensor()
    ])

#load the data into Torch
root1 = root + '/pictures'
DB = Reader.CustomImageDataset(annotations_file='Instadata.csv', img_dir= root1, transform= train_trans)
print(len(DB))
mnist_train, mnist_val = random_split(DB, [50, 10])


model = LitModel()

trainer = pl.Trainer(accelerator= 'cpu', precision=16, limit_train_batches=0.5)
trainer.fit(model, mnist_train,mnist_val)