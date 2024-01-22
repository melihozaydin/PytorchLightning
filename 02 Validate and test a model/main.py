import os

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms

import lightning as L

# Load data sets
dataset_dir = os.getcwd() + "/00 Datasets"
os.makedirs(dataset_dir,exist_ok=True)

transform = transforms.ToTensor()
train_set = datasets.MNIST(root=dataset_dir, download=True, train=True, transform=transform)
test_set = datasets.MNIST(root=dataset_dir, download=True, train=False, transform=transform)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

# model
from LitAutoEncoder import Encoder,Decoder
model = LitAutoEncoder(Encoder(), Decoder())

# initialize the Trainer
trainer = L.Trainer(max_epochs=20)

from torch.utils.data import DataLoader

batch_size = 1024

train_loader = DataLoader(train_set, batch_size=batch_size)
valid_loader = DataLoader(valid_set, batch_size=batch_size)

# train with both splits
trainer = L.Trainer(max_epochs=20)
trainer.fit(model, train_loader, valid_loader)

# test the model
from torch.utils.data import DataLoader
trainer.test(model, \
             dataloaders=DataLoader(test_set, batch_size=batch_size)
             )
