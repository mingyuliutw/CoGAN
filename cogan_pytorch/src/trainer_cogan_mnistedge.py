from torch.autograd import Variable
from net_cogan_mnistedge import *
import torch
import torch.nn as nn
import numpy as np
from init import *


class MNISTEdgeCoGanTrainer(object):
    def __init__(self, batch_size=64, latent_dims=100):
        super(MNISTEdgeCoGanTrainer, self).__init__()
        self.dis = CoDis28x28()
        self.gen = CoGen28x28(latent_dims)
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0005)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0005)
        self.true_labels = Variable(torch.LongTensor(np.ones(batch_size*2, dtype=np.int)))
        self.fake_labels = Variable(torch.LongTensor(np.zeros(batch_size*2, dtype=np.int)))
        self.dis.apply(xavier_weights_init)
        self.gen.apply(xavier_weights_init)

    def cuda(self):
        self.dis.cuda()
        self.gen.cuda()
        self.true_labels = self.true_labels.cuda()
        self.fake_labels = self.fake_labels.cuda()

    def dis_update(self, images_a, images_b, noise):
        self.dis.zero_grad()
        true_outputs = self.dis(images_a, images_b)
        true_loss = nn.functional.cross_entropy(true_outputs, self.true_labels)
        _, true_predicts = torch.max(true_outputs.data, 1)
        true_acc = (true_predicts == 1).sum()/(1.0*true_predicts.size(0))
        fake_images_a, fake_images_b = self.gen(noise)
        fake_outputs = self.dis(fake_images_a, fake_images_b)
        fake_loss = nn.functional.cross_entropy(fake_outputs, self.fake_labels)
        _, fake_predicts = torch.max(fake_outputs.data, 1)
        fake_acc = (fake_predicts == 0).sum() / (1.0 * fake_predicts.size(0))
        d_loss = true_loss + fake_loss
        d_loss.backward()
        self.dis_opt.step()
        return 0.5 * (true_acc + fake_acc)

    def gen_update(self, noise):
        self.gen.zero_grad()
        fake_images_a, fake_images_b = self.gen(noise)
        fake_outputs = self.dis(fake_images_a, fake_images_b)
        fake_loss = nn.functional.cross_entropy(fake_outputs, self.true_labels.cuda())
        fake_loss.backward()
        self.gen_opt.step()
        return fake_images_a, fake_images_b

