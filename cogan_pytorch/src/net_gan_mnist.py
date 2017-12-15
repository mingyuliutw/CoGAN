import torch
import torch.nn as nn


# Discriminator Model
class Dis28x28(nn.Module):
    def __init__(self):
        super(Dis28x28, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(50, 500, kernel_size=4, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(500, 2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        out = self.model(x)
        return out.squeeze()


# Generator Model
class Gen28x28(nn.Module):
    def __init__(self, latent_dims):
        super(Gen28x28, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dims, 1024, kernel_size=4, stride=1),
            nn.BatchNorm2d(1024, affine=False),
            nn.PReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512, affine=False),
            nn.PReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, affine=False),
            nn.PReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=6, stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)
        out = self.model(x)
        return out