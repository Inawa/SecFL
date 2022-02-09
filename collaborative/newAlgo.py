import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
import numpy as np
import argparse
from torchvision.utils import save_image
from Nets import MnistFunc,MnistGenrator,MnistClassify,MnistDiscriminator,NET
from testAcc import test_img,test_img_nets
from Samping import mnist_split_2_user, mnist_iid
from util import DatasetSplit, DatasetFormat

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--ex", type=float, default=0.9, help="adam: learning rate")
    parser.add_argument("--attack", type=bool, default=False)
    parser.add_argument("--cheat", type=bool, default=True)
    opt = parser.parse_args()
    print(opt)
    return opt



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = load_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    model = MnistFunc().to(device)
    classify = MnistClassify().to(device)
    gen = MnistGenrator().to(device)
    dis = MnistDiscriminator().to(device)

    lossFun = torch.nn.CrossEntropyLoss().to(device)
    BCE_loss = torch.nn.BCELoss().to(device)

    optimizer_E = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_C = torch.optim.Adam(classify.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(dis.parameters(), lr=args.lr)
    optimizer_G = torch.optim.Adam(gen.parameters(), lr=args.lr)

    # schedulerE = torch.optim.lr_scheduler.ExponentialLR(optimizer_E, gamma=0.9999)
    # schedulerC = torch.optim.lr_scheduler.ExponentialLR(optimizer_C, gamma=0.9999)
    # schedulerD = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.9999)
    # schedulerG = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.9999)

    trans_form = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset_train = datasets.MNIST("../../data/mnist/", train=True, download=True, transform=trans_form)
    dataset_test = datasets.MNIST("../../data/mnist/", train=False, download=True, transform=trans_form)

    dict_users = mnist_split_2_user(dataset_train)
    Dataset_victim = DatasetSplit(dataset_train, dict_users[0])
    Dataset_attack = DatasetSplit(dataset_train, dict_users[1])

    Dataloader_victim = DataLoader(Dataset_victim, batch_size=64, shuffle=True)
    Dataloader_attack = DataLoader(Dataset_attack, batch_size=64, shuffle=True)


    for epoch in range(2000):
        #####train victim######
        for i, (imgs, label) in enumerate(Dataloader_victim):
            #print(label)
            valid = torch.FloatTensor(imgs.shape[0], 1).fill_(0.9).to(device)
            fake = torch.FloatTensor(imgs.shape[0], 1).fill_(0.1).to(device)

            optimizer_D.zero_grad()
            z = torch.FloatTensor(np.random.laplace(0, 1, (imgs.shape[0], 20))).to(device)
            encode_imgs = model(imgs.to(device)).detach()
            D_loss = (BCE_loss(dis(encode_imgs), fake) + BCE_loss(dis(z), valid)) / 2
            D_loss.backward()
            optimizer_D.step()


            if i % 3 == 0:
                optimizer_E.zero_grad()
                optimizer_C.zero_grad()
                encoding = model(imgs.to(device))
                classify_imgs = classify(encoding)
                C_loss = args.ex * lossFun(classify_imgs, label.to(device)) + (1-args.ex) * BCE_loss(dis(encoding), valid)
                C_loss.backward()
                optimizer_E.step()
                optimizer_C.step()


        if args.attack is True:
            gen.train()
            print("gen")
            for i in range(20):
                optimizer_G.zero_grad()
                z = torch.FloatTensor(np.random.normal(0, 1, (64, 100))).to(device)
                gen_imgs = gen(z)
                label_ = torch.LongTensor(64).fill_(3).to(device)
                out_ = model(gen_imgs)
                out_ = classify(out_)  ##大坑
                loss2 = lossFun(out_, label_)
                loss2.backward()
                optimizer_G.step()

            gen.eval()
            if args.cheat is True:
                print("cheat")
                z = torch.FloatTensor(np.random.normal(0, 1, (1000, 100))).to(device)
                gen_imgs = gen(z)
                label_f = torch.LongTensor(1000).fill_(10).to(device)
                gen_dataset = DatasetFormat(gen_imgs.detach().float().cpu(), label_f.cpu())
                Dataset_attack_tmp = gen_dataset + Dataset_attack
                Dataloader_attack = DataLoader(Dataset_attack_tmp, batch_size=64, shuffle=True)

        #####train attack#####
        #####train victim######
        for i, (imgs, label) in enumerate(Dataloader_attack):
            #print(label)
            valid = torch.FloatTensor(imgs.shape[0], 1).fill_(0.9).to(device)
            fake = torch.FloatTensor(imgs.shape[0], 1).fill_(0.1).to(device)

            optimizer_D.zero_grad()
            z = torch.FloatTensor(np.random.laplace(0, 1, (imgs.shape[0], 20))).to(device)
            encode_imgs = model(imgs.to(device)).detach()
            D_loss = (BCE_loss(dis(encode_imgs), fake) + BCE_loss(dis(z), valid)) / 2
            D_loss.backward()
            optimizer_D.step()

            if i % 3 == 0:
                optimizer_E.zero_grad()
                optimizer_C.zero_grad()
                encoding = model(imgs.to(device))
                classify_imgs = classify(encoding)
                C_loss = args.ex * lossFun(classify_imgs, label.to(device)) + (1-args.ex) * BCE_loss(dis(encoding), valid)
                C_loss.backward()
                optimizer_E.step()
                optimizer_C.step()


        # schedulerE.step()
        # schedulerC.step()
        # schedulerD.step()
        # schedulerG.step()

        ### 计算acc###
        nets = NET(model,classify,dis)
        acc_test, losstest = test_img_nets(nets, dataset_test, args)
        print("acc:%f,closs:%f,dloss:%f,gloss:%f" % (acc_test, C_loss, D_loss,0))

        if args.attack:
            os.makedirs("./new%f"%(args.ex), exist_ok=True)
            z = torch.FloatTensor(np.random.normal(0, 1, (64, 100))).to(device)
            gen_imgs1 = gen(z)
            save_image(gen_imgs1, "./new%f/epoch%d_%f.png" % (args.ex,epoch,acc_test), nrow=8)












