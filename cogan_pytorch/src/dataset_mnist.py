from __future__ import print_function
from PIL import Image
import cv2
import torch
import os
import numpy as np
import cPickle
import gzip
import torch.utils.data as data
import urllib


class MNISTSAMPLE(data.Dataset):

    def __init__(self, root, num_training_samples, train=True, transform=None, seed=None):
        self.url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        self.filename = 'mnist.pkl.gz'
        self.train = train
        self.root = root
        self.num_training_samples = num_training_samples
        self.transform = transform
        self.download()
        self.test_set_size = 0
        self.train_data, self.train_labels = self.load_samples()
        if seed is not None:
            np.random.seed(seed)
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.num_training_samples], ::]
            self.train_labels = self.train_labels[indices[0:self.num_training_samples]]
        self.train_data *= 255.0
        self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label)])
        return img, label

    def __len__(self):
        if self.train:
            return self.num_training_samples
        else:
            return self.test_set_size

    def download(self):
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, filename))
        urllib.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        if self.train:
            images = np.concatenate((train_set[0], valid_set[0]), axis=0)
            labels = np.concatenate((train_set[1], valid_set[1]), axis=0)
        else:
            images = test_set[0]
            labels = test_set[1]
            self.test_set_size = labels.shape[0]
        images = images.reshape((images.shape[0], 1, 28, 28))
        return images, labels

