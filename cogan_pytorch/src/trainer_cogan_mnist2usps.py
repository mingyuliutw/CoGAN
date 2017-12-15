from torch.autograd import Variable
from net_cogan_mnist2usps import *
import torch
import torch.nn as nn
import numpy as np
from init import *


class MNIST2USPSCoGanTrainer(object):
    def __init__(self, batch_size=64, latent_dims=100):
        super(MNIST2USPSCoGanTrainer, self).__init__()
        self.dis = CoDis28x28()
        self.gen = CoGen28x28(latent_dims)
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0005)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0005)
        self.true_labels = Variable(torch.LongTensor(np.ones(batch_size*2, dtype=np.int)))
        self.fake_labels = Variable(torch.LongTensor(np.zeros(batch_size*2, dtype=np.int)))
        self.dis.apply(xavier_weights_init)
        self.gen.apply(xavier_weights_init)
        self.mse_loss_criterion = torch.nn.MSELoss()

    def cuda(self):
        self.dis.cuda()
        self.gen.cuda()
        self.true_labels = self.true_labels.cuda()
        self.fake_labels = self.fake_labels.cuda()

    def dis_update(self, images_a, labels_a, images_b, noise, mse_wei, cls_wei):
        self.dis.zero_grad()
        # Adversarial part for true images
        true_outputs, true_feat_a, true_feat_b = self.dis(images_a, images_b)
        true_loss = nn.functional.cross_entropy(true_outputs, self.true_labels)
        _, true_predicts = torch.max(true_outputs.data, 1)
        true_acc = (true_predicts == 1).sum()/(1.0*true_predicts.size(0))

        # Adversarial part for fake images
        fake_images_a, fake_images_b = self.gen(noise)
        fake_outputs, fake_feat_a, fake_feat_b = self.dis(fake_images_a, fake_images_b)
        fake_loss = nn.functional.cross_entropy(fake_outputs, self.fake_labels)
        _, fake_predicts = torch.max(fake_outputs.data, 1)
        fake_acc = (fake_predicts == 0).sum() / (1.0 * fake_predicts.size(0))
        dummy_tensor = Variable(
            torch.zeros(fake_feat_a.size(0), fake_feat_a.size(1), fake_feat_a.size(2), fake_feat_a.size(3))).cuda()
        mse_loss = self.mse_loss_criterion(fake_feat_a - fake_feat_b, dummy_tensor) * fake_feat_a.size(
            1) * fake_feat_a.size(2) * fake_feat_a.size(3)

        # Classification loss
        cls_outputs = self.dis.classify_a(images_a)
        cls_loss = nn.functional.cross_entropy(cls_outputs, labels_a)
        _, cls_predicts = torch.max(cls_outputs.data, 1)
        cls_acc = (cls_predicts == labels_a.data).sum() / (1.0 * cls_predicts.size(0))

        d_loss = true_loss + fake_loss + mse_wei * mse_loss + cls_wei * cls_loss
        d_loss.backward()
        self.dis_opt.step()
        return 0.5 * (true_acc + fake_acc), mse_loss, cls_acc

    def gen_update(self, noise):
        self.gen.zero_grad()
        fake_images_a, fake_images_b = self.gen(noise)
        fake_outputs, fake_feat_a, fake_feat_b = self.dis(fake_images_a, fake_images_b)
        fake_loss = nn.functional.cross_entropy(fake_outputs, self.true_labels.cuda())
        fake_loss.backward()
        self.gen_opt.step()
        return fake_images_a, fake_images_b

