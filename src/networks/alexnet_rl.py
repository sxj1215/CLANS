import sys
import torch
import numpy as np
from src import utils
import copy
import math
from torch.nn import functional as F

class Net(torch.nn.Module):

    def __init__(self,conv, fc, last, embedding, scale, inputsize, taskcla, nlayers=2, nhid=2000,
                 pdrop1=0.2,
                 pdrop2=0.5, t=0):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        #self.c1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        #self.c2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        #self.c3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        #self.fc1=torch.nn.Linear(256*self.smid*self.smid,2048)
        #self.fc2=torch.nn.Linear(2048,2048)

        if t==0:# 初始化 卷积层为(卷积核个数，输入通道数，长，宽)
            self.c1w=torch.empty((64,ncha,size//8,size//8))
            self.c1b=torch.empty((64,))
            self.c1w,self.c1b=self.initialize(self.c1w,self.c1b)
            self.c1w=torch.nn.Parameter(self.c1w)
            self.c1b = torch.nn.Parameter(self.c1b)
            
            self.c2w =torch.empty((128,64,size//10,size//10))
            self.c2b =torch.empty((128,))
            self.c2w, self.c2b = self.initialize(self.c2w, self.c2b)
            self.c2w = torch.nn.Parameter(self.c2w)
            self.c2b = torch.nn.Parameter(self.c2b)
            
            self.c3w=torch.empty((256,128,2,2))
            self.c3b=torch.empty((256,))
            self.c3w, self.c3b = self.initialize(self.c3w, self.c3b)
            self.c3w = torch.nn.Parameter(self.c3w)
            self.c3b = torch.nn.Parameter(self.c3b)
            
            self.fc1=torch.nn.ParameterList()
            w_f= torch.empty((2048,256*self.smid*self.smid))
            b_f = torch.empty((2048,))
            w_f,b_f=self.initialize(w_f,b_f)
            self.fc1.append(torch.nn.Parameter(w_f))
            self.fc1.append(torch.nn.Parameter(b_f))
            
            #self.fc2 = torch.nn.Linear(2048, 2048)
            self.fc2 = torch.nn.ParameterList()
            w_f = torch.empty((2048, 2048))
            b_f = torch.empty((2048,))
            w_f, b_f = self.initialize(w_f, b_f)
            self.fc2.append(torch.nn.Parameter(w_f))
            self.fc2.append(torch.nn.Parameter(b_f))

            self.last=torch.nn.ModuleList()
            '''for t_, n in self.taskcla:
                self.last.append(torch.nn.Linear(2048, n))'''
            self.last = torch.nn.ParameterList()
            for t_, n in self.taskcla:
                w_l = torch.empty((n, 2048))
                b_l = torch.empty((n,))
                w_l, b_l = self.initialize(w_l, b_l)
                self.last.append(torch.nn.Parameter(w_l))
                self.last.append(torch.nn.Parameter(b_l))

            self.ec1 = torch.nn.Parameter(torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), 64), mean=0, std=1))
            self.ec2 = torch.nn.Parameter(torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), 128), mean=0, std=1))
            self.ec3 = torch.nn.Parameter(torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), 256), mean=0, std=1))
            self.efc1 = torch.nn.Parameter(torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), 2048), mean=0, std=1))
            self.efc2 = torch.nn.Parameter(torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), 2048), mean=0, std=1))
            #self.efc1 = torch.nn.Embedding(len(self.taskcla), 2048)
            #self.efc2 = torch.nn.Embedding(len(self.taskcla), 2048)
        else:# 继承旧模型并扩展
            c1w_exp = torch.empty(scale[0],ncha,size//8,size//8)
            c1b_exp = torch.empty((scale[0],))
            c1w_exp, c1b1_exp = self.initialize(c1w_exp, c1b_exp)
            self.c1w =torch.cat((conv['c1w'],c1w_exp.cuda()),dim=0)
            self.c1b =torch.cat((conv['c1b'],c1b_exp.cuda()),dim=0)
            self.c1w = torch.nn.Parameter(self.c1w)
            self.c1b = torch.nn.Parameter(self.c1b)

            n2_exp=torch.empty(conv['c2w'].shape[0], scale[0], size//10, size//10)
            try:
                torch.nn.init.kaiming_uniform_(n2_exp, a=math.sqrt(5))
            except Exception:
                pass
            n2_exp = torch.cat((conv['c2w'], n2_exp.cuda()), dim=1)
            c2w_exp = torch.empty(scale[1], conv['c1w'].shape[0]+scale[0], size//10, size//10)
            c2b_exp = torch.empty((scale[1],))
            c2w_exp, c2b2_exp = self.initialize(c2w_exp, c2b_exp)
            self.c2w = torch.cat((n2_exp, c2w_exp.cuda()), dim=0)
            self.c2b = torch.cat((conv['c2b'], c2b_exp.cuda()), dim=0)
            self.c2w = torch.nn.Parameter(self.c2w)
            self.c2b = torch.nn.Parameter(self.c2b)

            n3_exp = torch.empty(conv['c3w'].shape[0], scale[1], 2,2)
            try:
                torch.nn.init.kaiming_uniform_(n3_exp, a=math.sqrt(5))
            except Exception:
                pass
            n3_exp = torch.cat((conv['c3w'], n3_exp.cuda()), dim=1)
            c3w_exp = torch.empty(scale[2],conv['c2w'].shape[0]+scale[1],2,2)
            c3b_exp = torch.empty((scale[2],))
            c3w_exp, c3b_exp = self.initialize(c3w_exp, c3b_exp)
            self.c3w =torch.cat((n3_exp,c3w_exp.cuda()),dim=0)
            self.c3b =torch.cat((conv['c3b'],c3b_exp.cuda()),dim=0)
            self.c3w = torch.nn.Parameter(self.c3w)
            self.c3b = torch.nn.Parameter(self.c3b)

            self.fc1 = torch.nn.ParameterList()
            fcs= copy.deepcopy(fc)
            fc1_exp = torch.empty((fcs['fc1.0'].shape[0], 2*2*(scale[2])))
            try:
                torch.nn.init.kaiming_uniform_(fc1_exp, a=math.sqrt(5))
            except Exception:
                pass
            fc1_b=fcs['fc1.1']
            self.fc1.append(torch.nn.Parameter(torch.cat((fcs['fc1.0'], fc1_exp.cuda()), 1)))
            self.fc1.append(torch.nn.Parameter(fc1_b))

            #self.fc2 = copy.deepcopy(fc['fc2'])
            self.fc2 = torch.nn.ParameterList()
            fc2_w = fcs['fc2.0']
            fc2_b = fcs['fc2.1']
            self.fc2.append(torch.nn.Parameter(fc2_w))
            self.fc2.append(torch.nn.Parameter(fc2_b))

            self.last = copy.deepcopy(last)
            '''self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(2048,n))'''
            #expand1 = torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), scale[0]), mean=0, std=1)
            expand1 = torch.full((len(self.taskcla), scale[0]), -6)
            self.ec1 = torch.nn.Parameter(torch.cat((copy.deepcopy(embedding['ec1']), expand1.cuda()), dim=1))
            #expand2 = torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), scale[1]), mean=0, std=1)
            expand2 = torch.full((len(self.taskcla), scale[1]), -6)
            self.ec2 = torch.nn.Parameter(torch.cat((copy.deepcopy(embedding['ec2']), expand2.cuda()), dim=1))
            #expand3 = torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), scale[2]), mean=0, std=1)
            expand3= torch.full((len(self.taskcla), scale[2]), -6)
            self.ec3 = torch.nn.Parameter(torch.cat((copy.deepcopy(embedding['ec3']), expand3.cuda()), dim=1))
            self.efc1 = torch.nn.Parameter(copy.deepcopy(embedding['efc1']))
            self.efc2 = torch.nn.Parameter(copy.deepcopy(embedding['efc2']))

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        '''self.ec1=torch.nn.Embedding(len(self.taskcla),64)
        self.ec2=torch.nn.Embedding(len(self.taskcla),128)
        self.ec3=torch.nn.Embedding(len(self.taskcla),256)
        self.efc1=torch.nn.Embedding(len(self.taskcla),2048)
        self.efc2=torch.nn.Embedding(len(self.taskcla),2048)'''



        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        #"""
        for n, p in self.named_parameters():
            p.requires_grad = True



        return

    def forward(self,t,x,s=1,draw=False):
        # Gates
        masks=self.mask(t,s=s)
        gc1,gc2,gc3,gfc1,gfc2=masks
        # Gated
        h=F.conv2d(x, self.c1w, self.c1b, stride=1)
        h=self.maxpool(self.drop1(self.relu(h)))
        h=h*gc1.view(1,-1,1,1).expand_as(h)

        h=F.conv2d(h, self.c2w, self.c2b, stride=1)
        h=self.maxpool(self.drop1(self.relu(h)))
        h=h*gc2.view(1,-1,1,1).expand_as(h)

        h=F.conv2d(h, self.c3w, self.c3b, stride=1)
        h=self.maxpool(self.drop2(self.relu(h)))
        h=h*gc3.view(1,-1,1,1).expand_as(h)

        h=h.view(x.size(0),-1)
        h = self.drop2(self.relu(torch.nn.functional.linear(h, self.fc1[0], self.fc1[1])))
        h=h*gfc1.expand_as(h)

        #h=self.drop2(self.relu(self.fc2(h)))
        h = self.drop2(self.relu(torch.nn.functional.linear(h, self.fc2[0], self.fc2[1])))
        h=h*gfc2.expand_as(h)

        y=[]
        for t_,_ in self.taskcla:
            y.append(torch.nn.functional.linear(h, self.last[t_ * 2], self.last[t_ * 2 + 1]))
            #y.append(self.last[i](h))
        return y,masks

    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1[t])
        gc2=self.gate(s*self.ec2[t])
        gc3=self.gate(s*self.ec3[t])
        gfc1=self.gate(s*self.efc1[t])
        gfc2=self.gate(s*self.efc2[t])
        return [gc1,gc2,gc3,gfc1,gfc2]

    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gfc1,gfc2=masks
        if n=='fc1.0':
            post=gfc1.data.view(-1,1).expand_as(self.fc1[0])
            pre=gc3.data.view(-1,1,1).expand((self.ec3.shape[1],self.smid,self.smid)).contiguous().view(1,-1).expand_as(self.fc1[0])
            return torch.min(post,pre)
        elif n=='fc1.1':
            return gfc1.data.view(-1)
        elif n=='fc2.0':
            post=gfc2.data.view(-1,1).expand_as(self.fc2[0])
            pre=gfc1.data.view(1,-1).expand_as(self.fc2[0])
            return torch.min(post,pre)
        elif n=='fc2.1':
            return gfc2.data.view(-1)
        elif n=='c1w':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1w)
        elif n=='c1b':
            return gc1.data.view(-1)
        elif n=='c2w':
            post=gc2.data.view(-1,1,1,1).expand_as(self.c2w)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2w)
            return torch.min(post,pre)
        elif n=='c2b':
            return gc2.data.view(-1)
        elif n=='c3w':
            post=gc3.data.view(-1,1,1,1).expand_as(self.c3w)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.c3w)
            return torch.min(post,pre)
        elif n=='c3b':
            return gc3.data.view(-1)
        return None

    def initialize(self,w,b):
        try:
            torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(b, -bound, bound)
        except Exception:pass
        return w,b
