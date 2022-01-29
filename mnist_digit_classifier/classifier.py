import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch import profiler
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

dttm_start = pd.Timestamp.now()

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

batch_size = 2048

train_dataloader = DataLoader(training_data,
                              num_workers=4,
                              batch_size=batch_size)
validate_dataloader = DataLoader(validation_data,
                                 num_workers=4,
                                 batch_size=batch_size)
test_dataloader = DataLoader(test_data,
                             num_workers=4,
                             batch_size=batch_size)


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


max_epochs = 5
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpu_count, strategy='dp')
else:
    trainer = pl.Trainer(max_epochs=max_epochs)
model = MNISTClassifier()
trainer.fit(model, train_dataloader=train_dataloader)

dttm_finish = pd.Timestamp.now()

print(dttm_finish-dttm_start)