from __future__ import print_function
from PIL import Image
import cv2
import os
import numpy as np
import cPickle
import gzip
import torch.utils.data as data
import urllib


class MNISTEDGE(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        self.filename = 'mnist.pkl.gz'
        self.filename_train_domain_1 = "mnist_train_original.pickle"
        self.filename_train_domain_2 = "mnist_train_edge.pickle"
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.download()
        self.create_two_domains()
        # now load the picked numpy arrays
        if self.train:
            filename_train_domain_1 = os.path.join(self.root, self.filename_train_domain_1)
            filename_train_domain_2 = os.path.join(self.root, self.filename_train_domain_2)
            data_a, labels_a = cPickle.load(gzip.open(filename_train_domain_1, 'rb'))
            data_b, labels_b = cPickle.load(gzip.open(filename_train_domain_2, 'rb'))
            self.train_data_a = data_a * 255.0
            self.train_labels_a = labels_a
            self.train_data_b = data_b * 255.0
            self.train_labels_b = labels_b
            self.train_data_a = self.train_data_a.transpose((0, 2, 3, 1))  # convert to HWC
            self.train_data_b = self.train_data_b.transpose((0, 2, 3, 1))  # convert to HWC
            print(self.train_data_a.shape)
            print(self.train_data_b.shape)

    def __getitem__(self, index):
        index_2 = np.random.randint(0, self.__len__(), 1)
        if self.train:
            img_a, img_b = self.train_data_a[index, ::], self.train_data_b[index_2, ::].squeeze(axis=0)
        else:
            return
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return img_a, img_b

    def __len__(self):
        if self.train:
            return 30000
        else:
            return 0

    def download(self):
        filename = os.path.join(self.root, self.filename)
        if os.path.isfile(filename):
            return
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        print("Download %s to %s" % (self.url, filename))
        urllib.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def create_two_domains(self):

        def save_domains(input_data, input_labels, domain_1_filename, domain_2_filename):
            n_samples = input_data.shape[0]
            arr = np.arange(n_samples)
            np.random.shuffle(arr)
            data_a = np.zeros((n_samples // 2, 1, 28, 28))
            label_a = np.zeros(n_samples // 2, dtype=np.int32)
            data_b = np.zeros((n_samples - n_samples // 2, 1, 28, 28))
            label_b = np.zeros(n_samples - n_samples // 2, dtype=np.int32)
            for i in range(0, n_samples // 2):
                data_a[i, 0, :, :] = input_data[arr[i], :].reshape(28, 28)
                label_a[i] = input_labels[arr[i]]
            for i in range(n_samples // 2, n_samples):
                img = input_data[arr[i], :].reshape(28, 28)
                dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
                edge = dilation - img
                data_b[i - n_samples // 2, 0, :, :] = edge
                label_b[i - n_samples // 2] = input_labels[arr[i]]
            with gzip.open(domain_1_filename, 'wb') as handle:
                cPickle.dump((data_a, label_a), handle)
            with gzip.open(domain_2_filename, 'wb') as handle:
                cPickle.dump((data_b, label_b), handle)

        filename = os.path.join(self.root, self.filename)
        filename_train_domain_1 = os.path.join(self.root, self.filename_train_domain_1)
        filename_train_domain_2 = os.path.join(self.root, self.filename_train_domain_2)
        if os.path.isfile(filename_train_domain_1) and os.path.isfile(filename_train_domain_2):
            return
        f = gzip.open(filename, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        images = np.concatenate((train_set[0], valid_set[0]), axis=0)
        labels = np.concatenate((train_set[1], valid_set[1]), axis=0)
        print("Compute edge images")
        print("Save origin to %s and edge to %s" % (filename_train_domain_1, filename_train_domain_2))
        save_domains(images, labels, filename_train_domain_1, filename_train_domain_2)
        print("[DONE]")


