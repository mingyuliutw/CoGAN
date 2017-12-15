from __future__ import print_function
from PIL import Image
import cv2
import os
import numpy as np
import cPickle
import gzip
import torch.utils.data as data
import torch
import urllib


class USPSSAMPLE(data.Dataset):
    # Num of Train = 7438, Num ot Test 1860
    def __init__(self, root, num_training_samples, train=True, transform=None, seed=None):
        self.filename = 'usps_28x28.pkl'
        self.train = train
        self.root = root
        self.num_training_samples = num_training_samples
        self.transform = transform
        self.test_set_size = 0
        # self.download()
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
        data_set = cPickle.load(f)
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.test_set_size = labels.shape[0]
        return images, labels

