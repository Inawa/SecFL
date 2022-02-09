#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import numpy as np

#根据分得的数据集下标在整个数据集中取得每个client的数据集
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

#同上，格式不一样
class DatasetFormat(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label.numpy().tolist()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        image, label = self.dataset[item], self.label[item]
        return image, label





class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, client_id=None):
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.client_id = client_id
        self.DataSet = DatasetSplit(dataset, idxs)


    def train_attack(self, net, gen,attack=False):
        
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        loss_func = nn.CrossEntropyLoss()
        optimizer_G = torch.optim.SGD(gen.parameters(), lr=self.args.glr)

        #client是否为攻击者
        #g_size = 64
        g_size = 64
        if attack:
            ##攻击者训练generator 10,120,3000
            for i in range(10):
                gen.train()
                net.eval()
                for i in range(120):
                    z = torch.FloatTensor(np.random.normal(0, 1, (g_size, 100))).to(self.args.device)
                    gen_imgs = gen(z)
                    #生成要攻击多标签
                    label_ = torch.LongTensor(g_size).fill_(self.args.atk_label).to(self.args.device)   
                    out_ = net(gen_imgs)
                    loss2 = loss_func(out_, label_)
                    optimizer_G.zero_grad()
                    loss2.backward()
                    optimizer_G.step()
            print(loss2.item())
            #是否要生成假标签数据
            if self.args.initiative:
                #print("initiative")
                gen.eval()
                z = torch.FloatTensor(np.random.normal(0, 1, (self.args.fak_num, 100))).to(self.args.device)
                gen_imgs = gen(z)
                #为生成数据打上假标签，并加入到本地训练集
                label_f = torch.LongTensor(self.args.fak_num).fill_(self.args.fak_label).to(self.args.device)
                gen_dataset = DatasetFormat(gen_imgs.detach().float().cpu(), label_f.cpu())

                self.DataSet += gen_dataset
                
                if self.args.iid:
                    self.DataSet = gen_dataset
                    

                
                self.ldr_train = DataLoader(self.DataSet, batch_size=self.args.local_bs, shuffle=True)

        #正常训练
        epoch_loss = []
        for iter in range(self.args.local_ep):
            net.train()
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if attack:
            return gen.state_dict(), net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        else:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)





