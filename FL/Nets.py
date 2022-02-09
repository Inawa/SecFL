import torch.nn as nn
import torch.nn.functional as F

#mnist 分类器
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

#mnist 生成器
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
        self.fc2 = nn.Linear(400,41)
        self.sof = nn.Softmax(dim=1)

    def forward(self, img):
        x = self.conv(img)
        x = x.view(img.size()[0], -1)
        x = self.fc1(x)
        x = self.Tanh(x)
        x = self.fc2(x)
        #x = self.sof(x)
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
            nn.ConvTranspose2d(64,1,4,2,1),  #(32-1)*2+0-2*1+4 = 64
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



#cifar10生成器
class Cifar10Genator(nn.Module):
    def __init__(self):
        super(Cifar10Genator, self).__init__()
        self.n_g_feature = 64
        self.gnet = nn.Sequential(
            # 输入大小 = (64, 1, 1)
            #有点像互相关的反操作，(x-4)/1=1-->x=4
            nn.ConvTranspose2d(100, 4 * self.n_g_feature, kernel_size=4,
            bias=False),
            nn.BatchNorm2d(4 * self.n_g_feature),
            nn.ReLU(),
            # 大小 = (256, 4, 4)
            #{x+2(填充)-4(核尺寸)+2(步长)}/2=4-->x=8
            nn.ConvTranspose2d(4 * self.n_g_feature, 2 * self.n_g_feature, kernel_size=4,
                    stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * self.n_g_feature),
            nn.ReLU(),
            # 大小 = (128, 8, 8)
            nn.ConvTranspose2d(2 * self.n_g_feature, self.n_g_feature, kernel_size=4,
                    stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_g_feature),
            nn.ReLU(),
            # 大小 = (64, 16, 16)
            nn.ConvTranspose2d(self.n_g_feature, 3, kernel_size=4,
                    stride=2, padding=1),
            nn.Sigmoid(),
            # 图片大小 = (3, 32, 32)
        )
    def forward(self, input):
        input = input.view(-1, 100,1,1)
        output = self.gnet(input)
        return output



#ResNet18 分类cifar10
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=11):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)

if __name__ == '__main__':
    ResNet18
