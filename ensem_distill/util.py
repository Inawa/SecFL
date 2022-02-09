import argparse
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import numpy as np
from PIL import Image

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class DatasetFormat(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label.numpy().tolist()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        image, label = self.dataset[item], self.label[item]
        return image, label


class MyATT(Dataset):
    def __init__(self, dataset):
        
        self.dataset_ = dataset
        self.targets = []
        for i,v in enumerate(dataset):
            self.targets.append(v[1])


    def __getitem__(self, index):
        
        return self.dataset_[index][0], self.dataset_[index][1]

    def __len__(self):
        return len(self.dataset_)
def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_bs', type=int, default=32, help="")
    parser.add_argument('--test_bs', type=int, default=64, help="")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--glr', type=float, default=0.0002, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--frac', type=float, default=1, help="")
    parser.add_argument('--num_users', type=int, default=10, help="")
    parser.add_argument('--epochs', type=int, default=500, help="")
    parser.add_argument('--gen_epochs', type=int, default=30, help="")
    parser.add_argument('--local_epoch', type=int, default=1, help="")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--noniid1', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--noniid2', action='store_true', help='whether i.i.d or not')
    parser.add_argument("--attack", action='store_true')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")

    parser.add_argument('--atk_client', type=int, default=0, help="attacking client")
    parser.add_argument('--atk_label', type=int, default=3, help="attacking label")
    parser.add_argument('--fak_label', type=int, default=9, help="fake label")
    parser.add_argument("--initiative", action='store_true')
    parser.add_argument('--fak_num', type=int, default=1000, help="")

    parser.add_argument('--lr_G', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_S', type=float, default=0.002, help='learning rate')

    parser.add_argument('--atk_store',type=str,default='atk_store')

    opt = parser.parse_args()
    print(opt)
    return opt


    #lr_G0.2
    #lr_S2e-3
    #local_bs10
    #lr0.1

class MyDataset(Dataset):
    def __init__(self, txt, transform=None):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            if len(words)>2:
                print('error MyDataset')
            imgs.append((words[0],int(words[1])))

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn)
        img = np.asarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)
