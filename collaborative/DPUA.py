import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from torchvision.utils import save_image
from Nets import MnistGenrator, MNIST
from Samping import mnist_split_2_user, mnist_iid
from util import DatasetSplit, DatasetFormat
from testAcc import test_img
import os

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=int, default=1)
    parser.add_argument("--cheat", type=int, default=1)
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    opt = parser.parse_args()
    print(opt)
    return opt

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = load_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    model = MNIST().to(device)
    gen = MnistGenrator().to(device)

    lossFun = torch.nn.CrossEntropyLoss().to(device)
    optimizer_m = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer_G = torch.optim.SGD(gen.parameters(), lr=0.001)

    schedulerm = torch.optim.lr_scheduler.ExponentialLR(optimizer_m, gamma=0.9999)
    schedulerG = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.9999)

    trans_form = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset_train = datasets.MNIST("/home/huanghong/data/mnist", train=True, download=True, transform=trans_form)
    dataset_test = datasets.MNIST("/home/huanghong/data/mnist", train=False, download=True, transform=trans_form)

    dict_users = mnist_split_2_user(dataset_train)
    Dataset_victim = DatasetSplit(dataset_train, dict_users[0])
    Dataset_attack = DatasetSplit(dataset_train, dict_users[1])

    Dataloader_victim = DataLoader(Dataset_victim, batch_size=64, shuffle=True)
    Dataloader_attack = DataLoader(Dataset_attack, batch_size=64, shuffle=True)

    for epoch in range(2000):
        model.train()
        #####train victim######
        for i,(imgs,label) in enumerate(Dataloader_victim):
            out = model(imgs.to(device))
            loss = lossFun(out, label.to(device))
            optimizer_m.zero_grad()
            loss.backward()
            optimizer_m.step()
        model.eval()
        if args.attack:
            gen.train()
            #print("gen")
            for i in range(20):
                optimizer_G.zero_grad()
                z = torch.FloatTensor(np.random.normal(0, 1, (64, 100))).to(device)
                gen_imgs = gen(z)
                label_ = torch.LongTensor(64).fill_(3).to(device)
                out_ = model(gen_imgs)
                loss2 = lossFun(out_, label_)
                loss2.backward()
                optimizer_G.step()

            gen.eval()
            if args.cheat:
                #print("cheat")
                z = torch.FloatTensor(np.random.normal(0, 1, (1000, 100))).to(device)
                gen_imgs = gen(z)
                label_f = torch.LongTensor(1000).fill_(10).to(device)
                gen_dataset = DatasetFormat(gen_imgs.detach().float().cpu(), label_f.cpu())
                Dataset_attack_tmp = gen_dataset+Dataset_attack
                Dataloader_attack = DataLoader(Dataset_attack_tmp, batch_size=64, shuffle=True)

        model.train()
        for i,(imgs,label) in enumerate(Dataloader_attack):
            out = model(imgs.to(device))
            loss = lossFun(out, label.to(device))
            optimizer_m.zero_grad()
            loss.backward()
            optimizer_m.step()
        model.eval()

        schedulerm.step()
        schedulerG.step()

        ### 计算acc###
        acc_test, losstest = test_img(model, dataset_test, args)
        #print("acc:%f,loss:%f"%(acc_test,loss))


        ## save images
        if args.attack:
            gen.eval()
            os.makedirs("./recon4", exist_ok=True)
            z = torch.FloatTensor(np.random.normal(0, 1, (64, 100))).to(device)
            gen_imgs1 = gen(z)
            save_image(gen_imgs1, "./recon4/epoch%d_%f.png"%(epoch,acc_test), nrow=8)










