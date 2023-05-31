import copy
import sys
# import cupy as cp
import numpy as np
import torch
import math

from src import utils


class Net(torch.nn.Module):

    def __init__(self,weights, bias, last, embedding, scale, inputsize, taskcla, nlayers=2, nhid=500,
                 pdrop1=0.2,
                 pdrop2=0.5, t=0):
        super(Net, self).__init__()
        self.nhid = nhid
        self.ncha, self.size, _ = inputsize
        self.taskcla = taskcla

        self.nlayers = nlayers
        self.weights = copy.deepcopy(weights)
        self.bias = copy.deepcopy(bias)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(pdrop1)
        self.drop2 = torch.nn.Dropout(pdrop2)
        self.gate = torch.nn.Sigmoid()

        if t == 0:  # 初始化
            self.w1 = torch.empty((self.nhid, self.ncha * self.size * self.size))
            self.b1 = torch.empty((self.nhid,))
            self.w1,self.b1=self.initialize(self.w1,self.b1)
            # self.weights['w1'] = torch.normal(0, 0.08, size=(self.ncha * self.size * self.size, self.nhid))
            # self.b1 = torch.normal(0, 0.08, size=(self.nhid,))
            self.w1 = torch.nn.Parameter(self.w1)
            self.b1 = torch.nn.Parameter(self.b1)
            self.efc1 = torch.nn.Parameter(
                torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), self.nhid), mean=0, std=1))

            self.w2 = torch.empty((self.nhid , self.nhid))
            self.b2 = torch.empty((self.nhid ,))
            self.w2, self.b2 = self.initialize(self.w2, self.b2)
            # self.w2 = torch.normal(0, 0.08, size=(self.nhid, self.nhid))
            # self.b2 = torch.normal(0, 0.08, size=(self.nhid,)
            self.w2 = torch.nn.Parameter(self.w2)
            self.b2 = torch.nn.Parameter(self.b2)

            # self.efc1 = torch.nn.Parameter(torch.FloatTensor(len(self.taskcla), self.nhid).uniform_(0, 2))
            # self.efc2 = torch.nn.Parameter(torch.FloatTensor(len(self.taskcla), self.nhid-1).uniform_(0, 2))

            self.efc2 = torch.nn.Parameter(
                torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), self.nhid), mean=0, std=1))




        else:  # 继承旧模型并扩展
            '''if scale[0] == 0:  # 不扩展
                self.w1 = torch.nn.Parameter(self.weights['w1'])
                self.b1 = torch.nn.Parameter(self.bias['b1'])

                self.w2 = torch.nn.Parameter(self.weights['w2'])
                self.b2 = torch.nn.Parameter(self.bias['b2'])
                self.efc1 = torch.nn.Parameter(copy.deepcopy(embedding['efc1']))
                self.efc2 = torch.nn.Parameter(copy.deepcopy(embedding['efc2']))'''


            # 扩展网络
            # 扩展神经元个数
            w1_exp = torch.empty(scale[0], (self.weights['w1'].shape[1]))
            b1_exp = torch.empty((scale[0],))
            w1_exp,b1_exp=self.initialize(w1_exp,b1_exp)
            self.w1 = torch.cat((torch.tensor(self.weights['w1']).cuda(), w1_exp.cuda()), 0)
            self.b1 = torch.cat(((self.bias['b1'].cuda(), b1_exp.cuda())), 0)
            self.w1 = torch.nn.Parameter(self.w1)
            self.b1 = torch.nn.Parameter(self.b1)
            # self.fc1=torch.nn.Linear(self.ncha*self.size*self.size,nhid)
            # self.efc1=torch.nn.Embedding(len(self.taskcla),nhid)

            #expand1 = torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), scale[0]), mean=0, std=1)
            expand1 = torch.full((len(self.taskcla), scale[0]), -6)
            self.efc1 = torch.nn.Parameter(torch.cat((copy.deepcopy(embedding['efc1']), expand1.cuda()), dim=1))

            # 扩展神经元维度
            w2_exp = torch.empty((self.weights['w2'].shape[0], scale[0]))
            try:
                torch.nn.init.kaiming_uniform_(w2_exp, a=math.sqrt(5))
            except Exception:
                pass
            self.w2 = torch.cat(((self.weights['w2'].cuda(), w2_exp.cuda())), 1)

            # 扩展神经元个数
            n2_exp = torch.empty(scale[1], (self.w2.shape[1]))
            b2_exp = torch.empty((scale[1],))
            n2_exp,b2_exp=self.initialize(n2_exp,b2_exp)
            self.w2 = torch.cat((self.w2, n2_exp.cuda()), 0)
            self.b2 = torch.cat((self.bias['b2'], b2_exp.cuda()), 0)
            self.w2 = torch.nn.Parameter(self.w2)
            self.b2 = torch.nn.Parameter(self.b2)


            '''self.w2 = torch.nn.Parameter(self.w2)
            self.b2 = torch.nn.Parameter(self.bias['b2'])'''
            # self.fc2=torch.nn.Linear(nhid,nhid)
            # self.efc2=torch.nn.Embedding(len(self.taskcla),nhid)
            #expand2 = torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), scale[1]), mean=0, std=1)
            expand2 = torch.full((len(self.taskcla), scale[1]), -6)
            self.efc2 = torch.nn.Parameter(torch.cat((copy.deepcopy(embedding['efc2']), expand2.cuda()), dim=1))
            #self.efc2 = torch.nn.Parameter(copy.deepcopy(embedding['efc2']))

        if t == 0:
            self.last = torch.nn.ParameterList()
            for t_, n in self.taskcla:
                w_l= torch.empty((n,self.nhid))
                b_l = torch.empty((n,))
                w_l,b_l=self.initialize(w_l,b_l)
                self.last.append(torch.nn.Parameter(w_l))
                self.last.append(torch.nn.Parameter(b_l))
        else:
            self.last = copy.deepcopy(last)
            for t_, n in self.taskcla:
                # 扩展神经元维度
                w_exp = torch.empty((n, scale[1]))
                try:
                    torch.nn.init.kaiming_uniform_(w_exp, a=math.sqrt(5))
                except Exception:
                    pass
                #torch.nn.init.kaiming_uniform_(w_exp, a=math.sqrt(5))
                w_l = torch.cat((self.last[t_*2], w_exp.cuda()), 1)
                b_l=self.last[t_*2+1]
                self.last[t_*2]=torch.nn.Parameter(w_l)
                self.last[t_ * 2+1] = torch.nn.Parameter(b_l)

        for n, p in self.named_parameters():
            p.requires_grad = True
        # self.gate=torch.nn.Sigmoid()
        """ (e.g., used with compression experiments)
        lo,hi=0,2
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        self.efc3.weight.data.uniform_(lo,hi)
        """
        return

    def initialize(self,w,b):
        try:
            torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        except Exception:pass
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(b, -bound, bound)
        return w,b



    def forward(self, t, x, s=1):
        # Gates
        masks = self.mask(t, s=s)
        if self.nlayers == 1:
            gfc1 = masks
        elif self.nlayers == 2:
            gfc1, gfc2 = masks
        elif self.nlayers == 3:
            gfc1, gfc2, gfc3 = masks
        # Gated
        # print(gfc1,'*********************************\n',self.efc1.weight.T[0],'----------------------------------\n')
        h = self.drop1(x.view(x.size(0), -1))
        '''if self.o0 is None:
            self.o0=h
        else:
            self.o0 =torch.cat((self.o0,h),dim=0)'''
        h = self.drop2(self.relu(torch.nn.functional.linear(h, self.w1, self.b1)))

        h = h * gfc1.expand_as(h)

        if self.nlayers > 1:
            h = self.drop2(self.relu(torch.nn.functional.linear(h, self.w2, self.b2)))
            # self.o2.append(h)
            h = h * gfc2.expand_as(h)
            # self.o2_mask.append(h)
            if self.nlayers > 2:
                h = self.drop2(self.relu(torch.nn.functional.linear(h, self.w3, self.b3)))
                h = h * gfc3.expand_as(h)
        y = []
        # if t==0 and eval==False:
        for t_, i in self.taskcla:
            y.append(torch.nn.functional.linear(h, self.last[t_*2], self.last[t_*2+1]))
        '''else:
            for t_, i in self.taskcla:
                y.append(torch.nn.functional.linear(h, self.last[2*t_], self.last[2*t_+1]))'''
        return y, masks

    '''def transfer(self, embedding, cur_t, sim_t, neural):
        # 对相似任务的embedding进行操作 neural代表受控神经元的索引
        for n in range(len(embedding)):
            if n == 0:
                index1 = []
                index2 = []
                a = self.efc1.data
                for i in range(neural[n][sim_t]):
                    index1.append(i)
                for j in range(embedding[n].shape[1] - neural[n][sim_t]):
                    index2.append(neural[n][sim_t] + j)
                x = torch.gather(a[sim_t], 0, torch.LongTensor(index1).cuda())
                y = torch.gather(a[cur_t], 0, torch.LongTensor(index2).cuda())
                #print(a[sim_t])
                a[sim_t] = torch.cat((x, y), dim=0)
                self.efc1 = torch.nn.Parameter(a)
                #print(self.efc1.data[sim_t])'''

    def mask(self, t, s=1):
        gfc1 = self.gate(s * self.efc1[t])
        if self.nlayers == 1: return gfc1
        gfc2 = self.gate(s * self.efc2[t])
        if self.nlayers == 2: return [gfc1, gfc2]
        gfc3 = self.gate(s * self.efc3[t])
        return [gfc1, gfc2, gfc3]

    def get_view_for(self, n, masks):
        if self.nlayers == 1:
            gfc1 = masks
        elif self.nlayers == 2:
            gfc1, gfc2 = masks
        elif self.nlayers == 3:
            gfc1, gfc2, gfc3 = masks
        if n == 'w1':
            return gfc1.data.view(-1, 1).expand_as(self.w1)
        elif n == 'b1':
            return gfc1.data.view(-1)
        elif n == 'w2':
            post = gfc2.data.view(-1, 1).expand_as(self.w2)
            pre = gfc1.data.view(1, -1).expand_as(self.w2)
            return torch.min(post, pre)
        elif n == 'b2':
            return gfc2.data.view(-1)
        elif n == 'w3':
            post = gfc3.data.view(-1, 1).expand_as(self.w3)
            pre = gfc2.data.view(1, -1).expand_as(self.w3)
            return torch.min(post, pre)
        elif n == 'b3':
            return gfc3.data.view(-1)
        return None
