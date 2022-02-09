import torch
from nets import MnistGenrator
import numpy as np
from torchvision.utils import save_image
import copy
import numpy as np
from torch.nn.modules import loss
from torchvision import datasets, transforms
import torch
import os
from torch.utils.data import DataLoader, Dataset, TensorDataset
from nets import MnistGenrator, MNIST, ATTGenrator, ATT

z1 = torch.FloatTensor(np.random.normal(0, 1, (1, 100))).to("cuda:1")
gen = MnistGenrator().to("cuda:1")

for i in range(10):
    gen_w = torch.load('./noniid2/models/gen_mode_atk%d_epoch%d'%(i,100))
    gen.load_state_dict(gen_w)
    gen.eval()
    img = gen(z1)
    save_image(img, "./noniid2/images/atk%d.png"%i, nrow=1)


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