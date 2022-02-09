import copy
import os
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from samping import mnist_iid, mnist_noniid1, mnist_noniid2, mnist_noniid11,mnist_noniid22
from util import load_args
from nets import MNIST, Generator,Generator1,MnistGenrator,ATT,ATTGenrator
from update import clientUpdate
from test import test_img
import torch.nn.functional as F
import torch.nn as nn
from server import serverUpdate
import matplotlib.pyplot as plt
from Fed import FedAvg
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from PIL import Image

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
def train():
    args = load_args()
    args.device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else 'cpu')

    
    if args.dataset == "mnist":
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.MNIST("/home/huanghong/data/mnist/", train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST("/home/huanghong/data/mnist/", train=False, download=True, transform=trans_mnist)

        #glob_net = MNIST().to(args.device)
        gen = MnistGenrator().to(args.device)
    elif args.dataset == "ATT":
        dataset_train = torch.load("../ATT/train_dataset.db")
        dataset_test = torch.load("../ATT/test_dataset.db")
        #glob_net = ATT().to(args.device)
        gen = ATTGenrator().to(args.device)


    if args.iid:
        dict_users = mnist_iid(dataset_train, args.num_users)
    elif args.noniid1:
        #dict_users, dict_label = mnist_noniid1(dataset_train, args.num_users)
        dict_users = mnist_noniid11(dataset_train, args.num_users)
    elif args.noniid2:
        #dict_users, dict_label = mnist_noniid2(dataset_train, args.num_users)
        #dict_users = mnist_noniid22(dataset_train, args.num_users)
        dict_users = torch.load("../FL/dict_users10")

    models = []
    
    for i in range(args.num_users):
        if args.dataset == "mnist":
            model = MNIST().to(args.device)
        else:
            model = ATT().to(args.device)
        models.append(model)

    if args.dataset == "mnist":
        student_model = MNIST().to(args.device)
        w_gloabl = student_model.state_dict()
        test_model = MNIST().to(args.device)
        gen = MnistGenrator().to(args.device)
        g_w = gen.state_dict()
        gen2 = Generator().to(args.device)
        gen2_w = gen2.state_dict()
    else:
        #student_model = ATT().to(args.device)
        #w_gloabl = student_model.state_dict()
        test_model = ATT().to(args.device)
        gen = ATTGenrator().to(args.device)
        g_w = gen.state_dict()
        #gen2 = Generator1().to(args.device)
        #gen2_w = gen2.state_dict()


    server = serverUpdate(args=args)

    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)


    for epoch in range(args.epochs):
        w_loacls = []
        for idx in idxs_users:
            #acc, test_loss = test_img(models[idx],dataset_test,args)
            #print("epoch: %d, client:%d,acc1:%f"%(epoch,idx,acc))
            print("---------------client %d------------------"%idx)
            local = clientUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], client_id=idx)
            if idx == args.atk_client and args.attack:
                print("attacker%d" % idx)
                g_w, w, train_loss = local.train(net=copy.deepcopy(models[idx].to(args.device)),gen=copy.deepcopy(gen).to(args.device),attack=True)
                gen.load_state_dict(g_w)
            else:
                w, train_loss = local.train(net=copy.deepcopy(models[idx].to(args.device)),gen=copy.deepcopy(gen).to(args.device),attack=False)
            #w_loacls.append(w)

            models[idx].load_state_dict(w)




        if args.attack:
            #torch.save(g_w,'./models/gen_mode_atk{}_epoch{}'.format(args.atk_label,epoch))
            gen.eval()
            z1 = torch.FloatTensor(np.random.normal(0, 1, (64, 100))).to(args.device)
            gen_imgs1 = gen(z1)
            z2 = torch.FloatTensor(np.random.normal(0, 1, (1, 100))).to(args.device)
            gen_imgs2 = gen(z2)
            os.makedirs(args.atk_store, exist_ok=True)
            save_image(gen_imgs1, "./%s/epoch%d.png" %(args.atk_store, epoch), nrow=8)
            save_image(gen_imgs2, "./%s/epoch_%d.png" %(args.atk_store, epoch), nrow=1)


        args.lr *= 0.99

        w_gloabl, gen2_w= server.distillation(models,dataset_test)
        #dised_model_w, gen2_w= server.distillation2(models,dataset_test,dataset_train)


        for i in range(len(models)):
            models[i].load_state_dict(w_gloabl)


        test_model.load_state_dict(w_gloabl)
        acc, test_loss = test_img(test_model,dataset_test,args)
        print("-----epoch: %d,acc:%f----test_loss:%f--"%(epoch,acc,test_loss))




if __name__ == '__main__':
    train()
    #tmp()