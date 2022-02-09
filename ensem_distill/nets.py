import torch
import torch.nn as nn
from torch.nn.modules.normalization import GroupNorm


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,32,5),  #(28-5)/1+1 = 24
            nn.Tanh(),
            nn.MaxPool2d(3),    #24/3 = 8
            nn.Conv2d(32,64,5),  #(8-5)/1+1 = 4
            nn.MaxPool2d(2),   #4/2 = 2
        )
        self.fc1 = nn.Linear(256, 200)
        self.Tanh = nn.Tanh()
        self.fc2 = nn.Linear(200,10)
        self.sofmax = nn.Softmax(dim=1)

    def forward(self, img,out_feature=False):
        x = self.conv(img)
        x = x.view(img.size()[0], -1)
        feature = x
        x = self.fc1(x)
        x = self.Tanh(x)
        x = self.fc2(x)
        #x = self.sofmax(x)
        if out_feature:
            return x, feature
        else:
            return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 28 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(1, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img



class MnistGenrator(nn.Module):
    def __init__(self):
        super(MnistGenrator, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4),  #(1-1)*1+4 = 4
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), #(4-1)*2+0-2*1+4 = 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),  #(8-1)*2+0-2*1+4 = 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(64,1,4,2,1),  #(16-1)*2+0-2*1+4 = 32
            nn.Tanh(),
        )

    def forward(self, input):
        input = input.view(-1, 100,1,1)
        out = self.block1(input)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out[:,:,2:30,2:30]
        return out


class ATT(nn.Module):
    def __init__(self):
        super(ATT, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,32,5),  #(64-5)/1+1 = 60
            nn.Tanh(),
            nn.MaxPool2d(3),    #60/3 = 20
            nn.Conv2d(32,64,5),  #(20-5)/1+1 = 16
            nn.Tanh(),
            nn.MaxPool2d(2),   #16/2 = 8
            nn.Conv2d(64,128,5),  #(8-5)/1+1 = 4
            nn.Tanh(),
            nn.MaxPool2d(2),   #4/2 = 2
        )
        self.fc1 = nn.Linear(512, 400)
        self.Tanh = nn.Tanh()
        self.fc2 = nn.Linear(400,40)
        self.sof = nn.Softmax(dim=1)

    def forward(self, img, out_feature=False):
        x = self.conv(img)
        x = x.view(img.size()[0], -1)
        feature = x
        x = self.fc1(x)
        x = self.Tanh(x)
        x = self.fc2(x)
        #x = self.sof(x)
        if out_feature :
            return x, feature
        else:
            return x

class ATTGenrator(nn.Module):
    def __init__(self):
        super(ATTGenrator, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4),  #(1-1)*1+4 = 4
            nn.ReLU(),
            )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1), #(4-1)*2+0-2*1+4 = 8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1),  #(8-1)*2+0-2*1+4 = 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),  #(16-1)*2+0-2*1+4 = 32
            nn.ReLU(),
        )
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(64,1,4,2,1),  #(16-1)*2+0-2*1+4 = 32
            nn.Tanh(),
        )

    def forward(self, input):
        input = input.view(-1, 100,1,1)
        out = self.block1(input)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        return out


class Generator1(nn.Module):
    def __init__(self):
        super(Generator1, self).__init__()

        self.init_size = 64 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(1, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img