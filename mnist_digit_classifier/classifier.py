import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

training_data, validation_data = random_split(
    training_data,
    [
        int(len(training_data)*0.8),
        len(training_data) - int(len(training_data)*0.8)
    ],
    torch.Generator().manual_seed(666)
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
validate_dataloader = DataLoader(validation_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")


class MNISTClassifier(pl.LightningModule):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.conv2d_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 8, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*22*22, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        logits = self.conv2d_stack(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


trainer = pl.Trainer(max_epochs=50)
model = MNISTClassifier()
trainer.fit(model, train_dataloader=train_dataloader)
