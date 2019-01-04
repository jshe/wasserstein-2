import torch
import math
import random
import utils
import numpy as np
import h5py
import os
import torch.utils.data as data

from torchvision import datasets
from torchvision import transforms

def get_loader(config):
    mnist_loader = get_mnist_loader(config, batch_size=config.batch_size)
    usps_loader = get_usps_loader(config, batch_size=config.batch_size)

    if config.direction == 'usps-mnist':
           return mnist_loader, usps_loader
    elif config.direction == 'mnist-usps':
           return usps_loader, mnist_loader

def get_mnist_loader(config, batch_size, train=True):
    tf = transforms.Compose([transforms.Scale(16),
                              transforms.CenterCrop(16),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    mnist = datasets.MNIST(root=config.mnist_path, train=train, download=True, transform=tf)
    return RealDataGenerator(torch.utils.data.DataLoader(dataset=mnist,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=4))

def get_usps_loader(config, batch_size, train=True):
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    usps = USPS(root=config.usps_path, train=train, transform=tf)
    return RealDataGenerator(torch.utils.data.DataLoader(dataset=usps,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=4))


def get_data(config, train=True):
    if train:
        if config.direction == 'usps-mnist':
               loader = get_usps_loader(config, batch_size=7291)
        elif config.direction == 'mnist-usps':
               loader = get_mnist_loader(config, batch_size=60000)
    else:
        if config.direction == 'usps-mnist':
               loader = get_mnist_loader(config, batch_size=2007, train=False)
        elif config.direction == 'mnist-usps':
               loader = get_usps_loader(config, batch_size=10000, train=False)
    return next(loader)
   


class RealDataGenerator(object):
    """samples from real data"""
    "superclass of all data. WARNING: doesn't raise StopIteration so it loops forever!"

    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(self.loader)
        self.data_len = len(self.loader)
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def get_batch(self):
        if (((self.count + 1) % self.data_len) == 0):
            del self.iterator
            self.iterator = iter(self.loader)
        self.count += 1
        return next(self.iterator)

    def float_tensor(self, batch):
        return torch.from_numpy(batch).type(torch.FloatTensor)

class USPS(data.Dataset):
    """USPS Dataset from: https://www.kaggle.com/bistaumanga/usps-dataset"""

    def __init__(self, root, train=True, transform=None):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.dataset_size = None

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], :]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data = self.train_data.reshape(-1, 16, 16, 1)

    def __getitem__(self, index):
        img, label = self.train_data[index, :], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        return img, label

    def __len__(self):
        return self.dataset_size

    def load_samples(self):
        path = os.path.join(self.root, './usps.h5')
        with h5py.File(path, 'r') as hf:
            if self.train:
                train = hf.get('train')
                images = train.get('data')[:]
                labels = train.get('target')[:]
                self.dataset_size = labels.shape[0]
            else:
                test = hf.get('test')
                images = test.get('data')[:]
                labels = test.get('target')[:]
                self.dataset_size = labels.shape[0]
        return images, labels
