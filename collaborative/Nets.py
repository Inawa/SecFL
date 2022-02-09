import torch.nn as nn

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
        self.fc2 = nn.Linear(200,11)
        self.sof = nn.Softmax(dim=1)

    def forward(self, img):
        x = self.conv(img)
        x = x.view(img.size()[0], -1)
        x = self.fc1(x)
        x = self.Tanh(x)
        x = self.fc2(x)
        #x = self.sof(x)
        return x

class MnistFunc(nn.Module):
    def __init__(self):
        super(MnistFunc, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,32,5),  #(28-5)/1+1 = 24
            nn.MaxPool2d(3),    #24/3 = 8
            nn.Conv2d(32,64,5),  #(8-5)/1+1 = 4
            nn.MaxPool2d(2),   #4/2 = 2
        )
        self.fc1 = nn.Linear(64*2*2,100)
        self.tan = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(100,20)

    def forward(self, img):
        x = self.conv(img)
        x = x.view(img.size()[0], -1)
        x = self.fc1(x)
        x = self.tan(x)
        x = self.fc2(x)
        return x

class MnistGenrator(nn.Module):
    def __init__(self):
        super(MnistGenrator, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4),  #(1-1)*1+4 = 4
            nn.ReLU(),
            )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), #(4-1)*2+0-2*1+4 = 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),  #(8-1)*2+0-2*1+4 = 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
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

class MnistClassify(nn.Module):
    def __init__(self):
        super(MnistClassify, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(20, 200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(200, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100,11),
        )

    def forward(self, z):
        img_flat = self.model(z)
        return img_flat

class MnistDiscriminator(nn.Module):
    def __init__(self):
        super(MnistDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(20, 200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(200, 400),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(400, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, z):
        validity = self.model(z)
        validity = self.sig(validity)
        return validity




class NET(object):
    def __init__(self, F=None, C=None, D=None, G=None):
        self.F_net = F
        self.C_net = C
        self.D_net = D
        self.G_net = G

    def train(self):
        self.F_net.train()
        self.C_net.train()
        self.D_net.train()
        if self.G_net:
            self.G_net.train()

