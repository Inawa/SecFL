#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

#iid数据集划分


def mnist_iid(dataset, num_users):
    print("mnist_iid")
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(
            all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid1(dataset, num_users=10):
    print("mnist_noniid1")
    #labels = dataset.targets.numpy()
    labels = np.array(dataset.targets)
    idxs = [i for i in range(len(dataset))]
    idxs = np.arange(len(dataset))
    idxs_labels = np.vstack((idxs, labels))

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    idxs_new = np.array([])

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_labels = {}

    for i in range(10):
        pos = idxs_labels[1].tolist().index(i)
        idxs_new = np.concatenate((idxs_new, idxs[pos:pos+5000]), axis=0)

    num_imgs = int(50000/num_users)

    for i in range(num_users):
        dict_users[i] = idxs_new[i*num_imgs:(i+1)*num_imgs].astype('int64')
        dict_labels[i] = labels[int(idxs_new[i*num_imgs])]

    for i in range(num_users):
        print(labels[dict_users[i].astype('int64')])
    print(dict_labels)
    return dict_users, dict_labels


def mnist_noniid2(dataset, num_users=10):
    print("mnist_noniid2")
    #labels = dataset.targets.numpy()
    labels = np.array(dataset.targets)
    idxs = [i for i in range(len(dataset))]
    idxs = np.arange(len(dataset))
    idxs_labels = np.vstack((idxs, labels))

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    idxs_new = np.array([])

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    dict_labels = {i: [] for i in range(num_users)}
    for i in range(10):
        pos = idxs_labels[1].tolist().index(i)
        idxs_new = np.concatenate(
            (idxs_new, idxs[pos:pos+5000]), axis=0).astype('int64')

    num_shards = num_users*2
    num_imgs = int(50000/num_shards)
    idx_shard = [i for i in range(num_shards)]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs_new[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
            dict_labels[i].append(labels[int(idxs_new[rand*num_imgs])])

    for i in range(num_users):
        print(labels[dict_users[i].astype('int64')])
    print(dict_labels)
    return dict_users, dict_labels


def mnist_split_2_user(dataset):
    print("mnist_split_2_user")
    #labels = dataset.targets.numpy()
    labels = np.array(dataset.targets)
    idxs = [i for i in range(len(dataset))]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    mid = idxs_labels[1].tolist().index(5)
    dict_users = {}
    dict_users[0] = idxs[0:mid]
    dict_users[1] = idxs[mid:-1]
    return dict_users


# fedavg中non-iid分配策略
def mnist_noniid22(dataset, num_users):
    print("mnist_noniid22")
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards = num_users*2
    num_imgs = int(len(dataset)/num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


# 分多个non-iid用户, 按标签排序0-9，再均分
def mnist_noniid11(dataset, num_users):
    print("mnist_noniid2")
    #labels = dataset.targets.numpy()
    labels = np.array(dataset.targets)
    idxs = [i for i in range(len(dataset))]
    idxs_labels = np.vstack((idxs, labels))

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    dict_users = {}
    start = 0
    size = int(len(dataset)/num_users)
    for i in range(num_users):
        dict_users[i] = idxs[start:start+size]
        start += size
    return dict_users


def mnist_noniid10(dataset, num_users=10):
    print("mnist_noniid10")
    #labels = dataset.targets.numpy()
    labels = np.array(dataset.targets)
    idxs = [i for i in range(len(dataset))]
    idxs_labels = np.vstack((idxs, labels))

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    dict_users = {}

    for i in range(10):
        pos = idxs_labels[1].tolist().index(i)
        dict_users[i] = idxs[pos:pos+5000]

    print(type(dict_users[0]))
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST("/home/huanghong/data/mnist", train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))
    num = 3
    d, l = mnist_noniid2(dataset_train, 20)
    print(l)
