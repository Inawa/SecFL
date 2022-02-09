import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import numpy as np
from util import DatasetSplit, DatasetFormat


class clientUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, client_id=None):
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.client_id = client_id
        self.DataSet = DatasetSplit(dataset, idxs)

    def train(self, net, gen,attack=False):
        net.train()
        gen.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        optimizer_G = torch.optim.SGD(gen.parameters(), lr=self.args.glr)
        loss_func = nn.CrossEntropyLoss().to(self.args.device)


        #client是否为攻击者
        if attack:
            ##攻击者训练generator
            for i in range(10):
                gen.train()
                net.eval()
                for i in range(120):
                    z = torch.FloatTensor(np.random.normal(0, 1, (64, 100))).to(self.args.device)
                    gen_imgs = gen(z)
                    #生成要攻击多标签
                    label_ = torch.LongTensor(64).fill_(self.args.atk_label).to(self.args.device)   
                    out_ = net(gen_imgs)
                    loss2 = loss_func(out_, label_)
                    optimizer_G.zero_grad()
                    loss2.backward()
                    optimizer_G.step()

            #是否要生成假标签数据
            if self.args.initiative:
                print("initiative")
                z = torch.FloatTensor(np.random.normal(0, 1, (self.args.fak_num, 100))).to(self.args.device)
                gen_imgs = gen(z)
                #为生成数据打上假标签，并加入到本地训练集
                label_f = torch.LongTensor(self.args.fak_num).fill_(self.args.fak_label).to(self.args.device)
                gen_dataset = DatasetFormat(gen_imgs.detach().float().cpu(), label_f.cpu())

                self.DataSet += gen_dataset
                if self.args.iid:
                   self.DataSet = gen_dataset
                self.ldr_train = DataLoader(self.DataSet, batch_size=self.args.local_bs, shuffle=True)

                


        epoch_loss = []
        epoch_loss = []
        for epoch in range(self.args.local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            #print("epoch:%d,epoch_loss:%f"%(epoch,sum(batch_loss) / len(batch_loss)))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        #return gen.state_dict(), net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        if attack:
            return gen.state_dict(), net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        else:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)