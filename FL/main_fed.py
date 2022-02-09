import copy
import numpy as np
from torch.autograd.grad_mode import F
from torch.functional import _index_tensor_with_indices_list
from torchvision import datasets, transforms
import torch
from options import args_parser
from Samping import mnist_iid,mnist_noniid1, mnist_noniid2, mnist_split_2_user, mnist_noniid11, mnist_noniid22, split_ATT
from Nets import MnistGenrator, MNIST, ResNet18, Cifar10Genator, ATTGenrator, ATT
from Update import LocalUpdate, DatasetSplit
from Fed import FedAvg
from test import test_img
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import ssl
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

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
def remove_labela(dataset, idx, a):
    for id in list(idx):
        if dataset[id][1]==a:
            idx.remove(id) 


def train_normal_attack():
    args = args_parser()
    args.device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else 'cpu')
    print(args)
    print(args.device)
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trans_cifar10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    if args.dataset == "mnist":
        dataset_train = datasets.MNIST("/home/huanghong/data/mnist", train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST("/home/huanghong/data/mnist", train=False, download=True, transform=trans_mnist)
        glob_net = MNIST().to(args.device)
        gen = MnistGenrator().to(args.device)
    elif args.dataset == "ATT":
        dataset_train = torch.load("../ATT/train_dataset.db")
        dataset_test = torch.load("../ATT/test_dataset.db")
        glob_net = ATT().to(args.device)
        gen = ATTGenrator().to(args.device)
    elif args.dataset == "CIFAR":
        dataset_train = datasets.CIFAR10("/home/huanghong/data/cifar", train=True, download=True, transform=trans_cifar10)
        dataset_test = datasets.CIFAR10("/home/huanghong/data/cifar", train=False, download=True, transform=trans_cifar10)
        glob_net = ResNet18().to(args.device)
        gen = Cifar10Genator().to(args.device)

    # trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
    # llll = [0 for i in range(10)]
    # ooo = 0
    # for batch_idx, (images, labels) in enumerate(trainloader):
    #     os.makedirs("cifar_tmp", exist_ok=True)
    #     #save_image(gen_imgs1, "./%s/epoch%d.png" %(args.atk_store, iter), nrow=8)
    #     llll[labels]+=1
    #     save_image(images, "./cifar_tmp/%d_%d.png" %(labels, llll[labels]), nrow=1)
    #     ooo += 1
    #     if ooo == 200 :
    #         return
    



    if args.iid:
        dict_users = mnist_iid(dataset_train,args.num_users)
        #remove_labela(dataset_train,dict_users[args.atk_client],args.atk_label)
    elif args.noniid1:
        #dict_users, dict_labels = mnist_noniid1(dataset_train,args.num_users)
        #dict_users = split_ATT(dataset_train,args.num_users)
        dict_users = mnist_noniid11(dataset_train,args.num_users)
    elif args.noniid2:
        dict_users, dict_labels = mnist_noniid2(dataset_train,args.num_users)
        #dict_users = mnist_noniid22(dataset_train,args.num_users)
        #torch.save(dict_users,"./dict_users10")
        #dict_users = torch.load("./dict_users10")

    for i in range(args.num_users):
        print(len(dict_users[i]))

    acc_test_list = []


    for iter in range(args.epochs):
        loss_locals = []
        w_locals = []
        ##随机选取一部分选客户端
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        idxs_users = [i for i in range(args.num_users)]

        flag = 0
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], client_id=idx)

            ##默认最后一个编号client为attack，，
            #if idx == args.num_users-1 and args.attack:
            if idx == args.atk_client and args.attack :
                print("attacker%d"%idx)
                flag = 1
                g_a, w, loss = local.train_attack(net=copy.deepcopy(glob_net).to(args.device),
                                                gen=copy.deepcopy(gen).to(args.device),attack=True)
                gen.load_state_dict(g_a)
                #torch.save(g_a,'./ATTModel/atk%d_epoch%d.pth'%(args.atk_label,iter))
            else:
                w, loss = local.train_attack(net=copy.deepcopy(glob_net).to(args.device),
                                                gen=copy.deepcopy(gen).to(args.device), attack=False)

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        
        args.lr *= 0.99
        args.glr *= 0.9999
        #每轮选取client数目
        print("w_locals size:%d"%len(w_locals))

        #fedavg
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        glob_net.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

        acc_test, losstest = test_img(glob_net, dataset_test, args)
        print("Testing accuracy: {:.3f}".format(acc_test))


        #选取client中存在攻击者
        if flag == 1:
            flag = 0
            gen.eval()
            z = torch.FloatTensor(np.random.normal(0, 1, (64, 100))).to(args.device)
            z_1 = torch.FloatTensor(np.random.normal(0, 1, (1, 100))).to(args.device)
            gen_imgs1 = gen(z)
            gen_imgs2 = gen(z_1)
            os.makedirs(args.atk_store, exist_ok=True)
            save_image(gen_imgs1, "./%s/epoch%d.png" %(args.atk_store, iter), nrow=8)
            save_image(gen_imgs2, "./%s/epoch_%d.png" %(args.atk_store, iter), nrow=1)



def tmp():

    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    dataset_train = datasets.MNIST("/home/huanghong/data/mnist", train=True, download=True, transform=trans_mnist)
    labels = np.array(dataset_train.targets)
    dict_users = torch.load("./dict_users10")
    for i in range(10):
        print(labels[dict_users[i].astype('int64')])


if __name__ == '__main__':

    train_normal_attack()
    #tmp()