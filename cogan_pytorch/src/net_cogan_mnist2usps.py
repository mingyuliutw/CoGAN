import torch
import torch.nn as nn


# Discriminator Model
class CoDis28x28(nn.Module):
    def __init__(self):
        super(CoDis28x28, self).__init__()
        self.conv0_a = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.conv0_b = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.pool0 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(50, 500, kernel_size=4, stride=1, padding=0)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(500, 2, kernel_size=1, stride=1, padding=0)
        self.conv_cl = nn.Conv2d(500, 10, kernel_size=1, stride=1, padding=0)

    def forward(self, x_a, x_b):
        h0_a = self.pool0(self.conv0_a(x_a))
        h0_b = self.pool0(self.conv0_b(x_b))
        h1_a = self.pool1(self.conv1(h0_a))
        h1_b = self.pool1(self.conv1(h0_b))
        h2_a = self.prelu2(self.conv2(h1_a))
        h2_b = self.prelu2(self.conv2(h1_b))
        h2 = torch.cat((h2_a, h2_b), 0)
        h3 = self.conv3(h2)
        return h3.squeeze(), h2_a, h2_b

    def classify_a(self, x_a):
        h0_a = self.pool0(self.conv0_a(x_a))
        h1_a = self.pool1(self.conv1(h0_a))
        h2_a = self.prelu2(self.conv2(h1_a))
        h3_a = self.conv_cl(h2_a)
        return h3_a.squeeze()

    def classify_b(self, x_b):
        h0_b = self.pool0(self.conv0_b(x_b))
        h1_b = self.pool1(self.conv1(h0_b))
        h2_b = self.prelu2(self.conv2(h1_b))
        h3_b = self.conv_cl(h2_b)
        return h3_b.squeeze()


# Generator Model
class CoGen28x28(nn.Module):
    def __init__(self, latent_dims):
        super(CoGen28x28, self).__init__()
        self.dconv0 = nn.ConvTranspose2d(latent_dims, 1024, kernel_size=4, stride=1)
        self.bn0 = nn.BatchNorm2d(1024, affine=False)
        self.prelu0 = nn.PReLU()
        self.dconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512, affine=False)
        self.prelu1 = nn.PReLU()
        self.dconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256, affine=False)
        self.prelu2 = nn.PReLU()
        self.dconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.prelu3 = nn.PReLU()
        self.dconv4_a = nn.ConvTranspose2d(128, 1, kernel_size=6, stride=1, padding=1)
        self.dconv4_b = nn.ConvTranspose2d(128, 1, kernel_size=6, stride=1, padding=1)
        self.sig4_a = nn.Sigmoid()
        self.sig4_b = nn.Sigmoid()

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        h0 = self.prelu0(self.bn0(self.dconv0(z)))
        h1 = self.prelu1(self.bn1(self.dconv1(h0)))
        h2 = self.prelu2(self.bn2(self.dconv2(h1)))
        h3 = self.prelu3(self.bn3(self.dconv3(h2)))
        out_a = self.sig4_a(self.dconv4_a(h3))
        out_b = self.sig4_b(self.dconv4_b(h3))
        return out_a, out_b

