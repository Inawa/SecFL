#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):

    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_data(dataset, num_users):
    labels = dataset.train_labels.numpy()
    idxs = [i for i in range(len(dataset))]
    idxs_labels = np.vstack((idxs, labels))

    index3 = np.where(labels==3)
    print(index3)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    dict_users = {}
    dict_users[0] = idxs[0:20000]
    dict_users[1] = idxs[20000:40000]
    dict_users[2] = idxs[40000:60000]

    return dict_users

def mnist_split_2_user(dataset):
    labels = dataset.targets.numpy()
    idxs = [i for i in range(len(dataset))]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    mid = idxs_labels[1].tolist().index(5)
    dict_users = {}
    dict_users[0] = idxs[0:mid]
    dict_users[1] = idxs[mid:-1]

    #print(len(dict_users[0]))
    #print(len(dict_users[1]))
    return dict_users



def mnist_data1(dataset, num_users):
    labels = dataset.targets.numpy()
    idxs = [i for i in range(len(dataset))]
    label3 = set(np.where(labels==3)[0])

    idxs = list(set(idxs)-label3)
    label3 = list(label3)

    num_item = int(len(idxs)/num_users)

    dict_users = {}
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(idxs, num_item, replace=False))
        idxs = list(set(idxs) - dict_users[i])

    num_item3 = int(len(label3)/(num_users-1))
    for j in range(num_users-1):
        tmp = set(np.random.choice(label3, num_item3, replace=False))
        dict_users[j] = dict_users[j].union(tmp)
        label3 = list(set(label3) - tmp)

    return dict_users



if __name__ == '__main__':
    dataset_train = datasets.MNIST('../../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    mnist_split_2_user(dataset_train)
