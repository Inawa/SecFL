import torch
from Nets import MnistGenrator
import numpy as np
from torchvision.utils import save_image
import copy
import numpy as np
from torch.nn.modules import loss
from torchvision import datasets, transforms
import torch
import os
from torch.utils.data import DataLoader, Dataset, TensorDataset
from Nets import MnistGenrator, MNIST, ResNet18, Cifar10Genator, ATTGenrator, ATT

z1 = torch.FloatTensor(np.random.normal(0, 1, (1, 100))).to("cuda:1")
gen = ATTGenrator().to("cuda:1")

gen_w = torch.load('./experiment_result_att/atk33_epoch432.pth')
gen.load_state_dict(gen_w)
gen.eval()
img = gen(z1)
save_image(img, "./experiment_result_att/atk33_432.png", nrow=1)


'''
def fun(str):
    list1 = torch.load("./experiment_result/%s.pth"%str)
    os.makedirs("./tmp/%s"%str, exist_ok=True)
    for i in range(10):
        save_image(list1[i],"./tmp/%s/%d.png"%(str,i),nrow =1)


fun("iidlist1")
fun("iidlist2")
fun("noniidlist1")
fun("noniidlist2")
fun("list5")
'''