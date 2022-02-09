import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import numpy as np
from util import DatasetSplit
import torch.nn.functional as F
from nets import Generator,MNIST,MnistGenrator, ATTGenrator,Generator1,ATT
from util import load_args

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl

def kdloss2(y, teacher_scores):
    p = torch.log(y)
    l_kl = F.kl_div(p, teacher_scores, size_average=False)  / y.shape[0]
    return l_kl


def models_fun(models,data,args):
    size = data.size(0)
    if args.dataset == "mnist":
        res = torch.zeros(size=[size,10]).to(torch.device(args.device))
        features = torch.zeros(size=[size,256]).to(torch.device(args.device))
    else:
        res = torch.zeros(size=[size,40]).to(torch.device(args.device))
        features = torch.zeros(size=[size,512]).to(torch.device(args.device))
    for model in models:
        log_probs,feature = model(data,out_feature=True)
        #log_probs = F.softmax(log_probs,dim=1)
        res += log_probs
        features += feature
    
    return res/len(models), features/len(models)

def models_fun3(models,data,args):
    size = data.size(0)
    res = torch.zeros(size=[size,10]).to(torch.device(args.device))
    for model in models:
        
        log_probs = model(data)
        log_probs = F.softmax(log_probs,dim=1)
        res += log_probs
    idx = res.max(1)[1]
    out = torch.FloatTensor(F.one_hot(idx,10).float().cpu())
    return out.to(torch.device(args.device))


class serverUpdate(object):
    def __init__(self, args, idxs=None):
        self.args = args

    def distillation(self, models, data_test):
        total_correct = 0

        print(len(models))
        with torch.no_grad():
            data_test_loader = DataLoader(data_test, batch_size=128, num_workers=0)
            for i, (images, labels) in enumerate(data_test_loader):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                output,feature = models_fun(models,images,self.args)
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

            print(float(total_correct) / len(data_test))
        

        if self.args.dataset == "mnist":
            student = MNIST().to(self.args.device)
            generator = Generator().to(self.args.device)
        else:
            student = ATT().to(self.args.device)
            generator = Generator1().to(self.args.device)

        #student.load_state_dict(w_global)
        #generator.load_state_dict(gen_w)
        

        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.args.lr_G)
        optimizer_S = torch.optim.Adam(student.parameters(), lr=self.args.lr_S)
        criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        
        schedulerS = torch.optim.lr_scheduler.ExponentialLR(optimizer_S, gamma=0.95)
        schedulerG = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.95)


        generator.train()
        for epoch in range(self.args.gen_epochs):
            total_correct = 0
            avg_loss = 0.0

            for i in range(120):
                student.train()
                #512
                z = torch.FloatTensor(np.random.normal(0, 1, (128, 100))).to(self.args.device)  #128,100
                optimizer_G.zero_grad()
                optimizer_S.zero_grad()        
                gen_imgs = generator(z)

                outputs_T, features_T = models_fun(models,gen_imgs,self.args)  

                #print(outputs_T[0])

                pred = outputs_T.data.max(1)[1]
                loss_activation = -features_T.abs().mean()

                loss_one_hot = criterion(outputs_T,pred)
                softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
                #softmax_o_T = outputs_T.mean(dim = 0)
                loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
                
                loss =  loss_information_entropy * 5 + loss_one_hot * 1 #+ loss_activation * 0.1
                student_out = student(gen_imgs.detach())
                loss_kd = kdloss(student_out, outputs_T.detach())

                loss += loss_kd  
                loss.backward()
                optimizer_G.step()
                optimizer_S.step() 
                if i==1:
                    print ("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_kd: %f]" % (epoch+1, self.args.gen_epochs,loss_one_hot.item(), loss_information_entropy.item(), loss_kd.item()))

                if abs(loss_kd.item()) < 0.0000001 and i==1:
                    print("---------------------kd_loss==0-----------------------")

            
            with torch.no_grad():
                data_test_loader = DataLoader(data_test, batch_size=128, num_workers=4)
                for i, (images, labels) in enumerate(data_test_loader):
                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)
                    student.eval()
                    output= student(images)
                    avg_loss += criterion(output, labels).sum()
                    pred = output.data.max(1)[1]
                    total_correct += pred.eq(labels.data.view_as(pred)).sum()
                print(float(total_correct) / len(data_test))
            
            schedulerS.step()
            schedulerG.step()

            

        student.eval()
        generator.eval()
        return student.state_dict(), generator.state_dict()
    

    #
    def distillation1(self, models, data_test, w_gloab):
        total_correct = 0
        with torch.no_grad():
            data_test_loader = DataLoader(data_test, batch_size=128, num_workers=0)
            for i, (images, labels) in enumerate(data_test_loader):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                output,feature = models_fun(models,images,self.args)
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

            print(float(total_correct) / len(data_test))
        
        student = MNIST().to(self.args.device)
        generator = Generator().to(self.args.device)

        #generator.load_state_dict(gen_w)

        #generator = MnistGenrator().to(self.args.device)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.args.lr_G)
        #optimizer_S = torch.optim.SGD(student.parameters(), lr=self.args.lr_S, momentum=0.9, weight_decay=5e-4)
        optimizer_S = torch.optim.Adam(student.parameters(), lr=self.args.lr_S)
        criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        
        schedulerS = torch.optim.lr_scheduler.ExponentialLR(optimizer_S, gamma=0.95)
        schedulerG = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.95)


        generator.train()
        for epoch in range(self.args.gen_epochs):
            total_correct = 0
            avg_loss = 0.0

            for i in range(120):
                student.train()
                z = torch.FloatTensor(np.random.normal(0, 1, (128, 100))).to(self.args.device)
                optimizer_G.zero_grad()
                optimizer_S.zero_grad()        
                gen_imgs = generator(z)

                outputs_T, features_T = models_fun(models,gen_imgs,self.args)  
                pred = outputs_T.data.max(1)[1]
                loss_activation = -features_T.abs().mean()

                loss_one_hot = criterion(outputs_T,pred)
                softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)

                loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
                
                loss =  loss_information_entropy * 5 + loss_one_hot * 1#+ loss_activation * 0.1
                student_out = student(gen_imgs.detach())
                loss_kd = kdloss(student_out, outputs_T.detach())

                loss += loss_kd  
                loss.backward()
                optimizer_G.step()
                optimizer_S.step() 
                if i==1:
                    print ("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_kd: %f]" % (epoch+1, 20,loss_one_hot.item(), loss_information_entropy.item(), loss_kd.item()))

                if abs(loss_kd.item()) < 0.0000001 and i==1:
                    print("---------------------kd_loss==0-----------------------")



            with torch.no_grad():
                data_test_loader = DataLoader(data_test, batch_size=128, num_workers=0)
                for i, (images, labels) in enumerate(data_test_loader):
                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)
                    student.eval()
                    output= student(images)
                    avg_loss += criterion(output, labels).sum()
                    pred = output.data.max(1)[1]
                    total_correct += pred.eq(labels.data.view_as(pred)).sum()

                print(float(total_correct) / len(data_test))
            
            schedulerS.step()
            schedulerG.step()

            

        student.eval()
        generator.eval()
        return student.state_dict(), generator.state_dict()


    #有数据distillation
    def distillation2(self, models, data_test, data_train):
        total_correct = 0
        with torch.no_grad():
            data_test_loader = DataLoader(data_test, batch_size=128, num_workers=0)
            for i, (images, labels) in enumerate(data_test_loader):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                output, feature = models_fun(models, images, self.args)
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

            print(float(total_correct) / len(data_test))
        
        student = MNIST().to(self.args.device)
        generator = Generator().to(self.args.device)

        #generator.load_state_dict(gen_w)

        optimizer_S = torch.optim.Adam(student.parameters(), lr=self.args.lr_S)
        criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        

        data_train_loader = DataLoader(data_train, batch_size=128, num_workers=0)
                
        for epoch in range(10):
            total_correct = 0
            avg_loss = 0.0

            for i, (images, labels) in enumerate(data_train_loader):
                images = images.to(self.args.device)
                labels = images.to(self.args.device)
                student.train()

                optimizer_S.zero_grad()        

                outputs_T, features_T = models_fun(models, images, self.args)
                pred = outputs_T.data.max(1)[1]

                loss_kd = kdloss2(student(images), outputs_T.detach())
                
                
                loss_kd.backward()
                optimizer_S.step() 
                if i==1:
                    print ("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_kd: %f]" % (epoch+1, 20,0, 0, loss_kd.item()))
    


            with torch.no_grad():
                data_test_loader = DataLoader(data_test, batch_size=128, num_workers=0)
                for i, (images, labels) in enumerate(data_test_loader):
                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)
                    student.eval()
                    output = student(images)
                    avg_loss += criterion(output, labels).sum()
                    pred = output.data.max(1)[1]
                    total_correct += pred.eq(labels.data.view_as(pred)).sum()

                print(float(total_correct) / len(data_test))

        student.eval()
        generator.eval()
        return student.state_dict(), generator.state_dict()




if __name__ == '__main__':
    args = load_args()
    args.device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else 'cpu')
    server = serverUpdate(args)
    z = torch.randn(128, 100).to(args.device)
    out = server.distillation(None)
