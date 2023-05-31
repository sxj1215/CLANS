import sys, time
import numpy as np
import torch
import math
from src import utils

'''import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE'''


########################################################################################################################

class Appr(object):

    def __init__(self, model, mask_pre, old_mask, nepochs, sbatch=64, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=10000, lamb=0.75, smax=400, args=None):
        self.model = model

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.ce = torch.nn.CrossEntropyLoss()

        self.lamb = lamb
        self.smax = smax
        if len(args.parameter) >= 1:
            params = args.parameter.split(',')
            print('Setting parameters to', params)
            self.lamb = float(params[0])
            self.smax = float(params[1])

        self.mask_pre = mask_pre
        self.old_mask = old_mask
        self.mask_back = None

        self.optimizer = self._get_optimizer()

        # self.loss_x = []
        return

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        return torch.optim.SGD(self.model.parameters(), lr=lr)

    def train(self, t, xtrain, ytrain, xvalid, yvalid):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0 = time.time()
                self.train_epoch(t, xtrain, ytrain)
                clock1 = time.time()
                _, train_loss, train_acc = self.eval(t, xtrain, ytrain)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e + 1,
                                                                                                            1000 * self.sbatch * (
                                                                                                                    clock1 - clock0) / xtrain.size(
                                                                                                                0),
                                                                                                            1000 * self.sbatch * (
                                                                                                                    clock2 - clock1) / xtrain.size(
                                                                                                                0),
                                                                                                            train_loss,
                                                                                                            100 * train_acc),
                      end='')
                # Valid
                _, valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
                # Adapt lr
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = utils.get_model(self.model)
                    patience = self.lr_patience
                    print(' *', end='')
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= self.lr_factor
                        print(' lr={:.1e}'.format(lr), end='')
                        if lr < self.lr_min:
                            print()
                            break
                        patience = self.lr_patience
                        self.optimizer = self._get_optimizer(lr)
                print()

        except KeyboardInterrupt:
            print()

        # Restore best validation model
        self.model = utils.set_model_(self.model, best_model)

    def epoch(self, t, xtrain, ytrain, xvalid, yvalid, e, hp, xbuf=None, ybuf=None):

        clock0 = time.time()
        self.train_epoch(t, xtrain, ytrain, xbuf, ybuf, hp)
        clock1 = time.time()
        _, train_loss, train_acc = self.eval(t, xtrain, ytrain)
        clock2 = time.time()
        print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e + 1,
                                                                                                    1000 * self.sbatch * (
                                                                                                            clock1 - clock0) / xtrain.size(
                                                                                                        0),
                                                                                                    1000 * self.sbatch * (
                                                                                                            clock2 - clock1) / xtrain.size(
                                                                                                        0),
                                                                                                    train_loss,
                                                                                                    100 * train_acc),
              end='')
        # Valid
        _, valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
        print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
        # Adapt lr
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_model = utils.get_model(self.model, 'bm')
            print(' *', end='')
        else:
            self.patience -= 1
            if self.patience <= 0:
                self.lr /= self.lr_factor
                print(' lr={:.1e}'.format(self.lr), end='')
                if self.lr < self.lr_min:
                    print('lr too low!')
                    return None
                self.patience = self.lr_patience
        print()
        return 0

    def create_mask_back(self):
        # Weights mask
        self.mask_back = {}
        for n, _ in self.model.named_parameters():
            vals = self.model.get_view_for(n, self.mask_pre)
            if vals is not None:
                self.mask_back[n] = 1 - vals

    def finish(self, t, xtrain, ytrain, xvalid, yvalid, clock0, e):

        clock1 = time.time()
        _, train_loss, train_acc = self.eval(t, xtrain, ytrain)
        clock2 = time.time()
        print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e + 1,
                                                                                                    1000 * self.sbatch * (
                                                                                                            clock1 - clock0) / xtrain.size(
                                                                                                        0),
                                                                                                    1000 * self.sbatch * (
                                                                                                            clock2 - clock1) / xtrain.size(
                                                                                                        0),
                                                                                                    train_loss,
                                                                                                    100 * train_acc),
              end='')
        # Valid
        _, valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
        print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
        # Adapt lr
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_model = utils.get_model(self.model)
            self.patience = self.lr_patience
            print(' *', end='')
        else:
            self.patience -= 1
            if self.patience <= 0:
                self.lr /= self.lr_factor
                print(' lr={:.1e}'.format(self.lr), end='')
                self.patience = self.lr_patience
                self.optimizer = self._get_optimizer(self.lr)
        if t == 0: print()

        return

    def train_epoch(self, t, x, y, thres_cosh=50, thres_emb=6):
        self.model.train()

        r = np.arange(x.size(0))

        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = torch.autograd.Variable(x[b], volatile=False)
            targets = torch.autograd.Variable(y[b], volatile=False)
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False)
            s = (self.smax - 1 / self.smax) * i / len(r) + 1 / self.smax

            # Forward
            outputs, masks = self.model.forward(task, images, s=s)
            output = outputs[t]
            loss, _ = self.criterion(output, targets, masks)
            # Backward
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            if t > 0:
                for n, p in self.model.named_parameters():
                    if n in self.mask_back and p.grad is not None:
                        p.grad.data *= self.mask_back[n]

            for n, p in self.model.named_parameters():
                if n.startswith('e'):
                    num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                    den = torch.cosh(p.data) + 1
                    p.grad.data *= self.smax / s * num / den

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            for n, p in self.model.named_parameters():
                if n.startswith('e'):
                    p.data = torch.clamp(p.data, -thres_emb, thres_emb)
        return

    def transfer(self, t, xbuf, ybuf, ne_max, valid_acc, taskWithSimilarity, thres_cosh=50, thres_emb=6, ):

        best_acc = valid_acc

        for n, p in self.model.named_parameters():
            if n.startswith('e') or n.startswith('l'):
                p.requires_grad = True
            else:
                p.requires_grad = False
        lr = 0.05
        op = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        for t_ in range(t):

            self.best_model = utils.get_model(self.model, 'bm')

            count = 0
            for n, p in self.model.named_parameters():
                if n.startswith('e'):

                    p.data[t_] = p.data[t_] * (1 - (
                            (1 - self.old_mask[t_][count]) * self.mask_pre[count])).round() + torch.nn.init.normal_(
                        torch.FloatTensor(p.data[t_].shape), mean=0, std=1).cuda() * (
                                         (1 - self.old_mask[t_][count]) * self.mask_pre[count]).round()

                    count += 1
                elif n.startswith('l'):
                    if float(n[5:]) / 2 == t_:  # last.n/n+1判断任务id
                        m = torch.unsqueeze(((1 - self.old_mask[t_][-1]) * self.mask_pre[-1]).round(), 0).expand_as(p)
                        reinit = torch.empty_like(p.data)
                        torch.nn.init.kaiming_uniform_(reinit, a=math.sqrt(5))
                        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(reinit)
                        p.data = p.data * (1 - m) + reinit.cuda() * m
                    elif float(n[5:]) / 2 == t_ + 0.5:  # last.b
                        b_back = p.data.detach().clone()
                        b = torch.empty_like(p.data)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        torch.nn.init.uniform_(b, -bound, bound)
                        p.data = b

            print('t{}:'.format(t_), end='')
            patience = 5
            v_best = np.inf

            for e in range(ne_max):
                self.model.train()

                d = np.arange(xbuf[t_].size(0))
                np.random.shuffle(d)
                d = torch.LongTensor(d).cuda()
                for i in range(0, len(d), self.sbatch):
                    if i + self.sbatch <= len(d):
                        b = d[i:i + self.sbatch]
                    else:
                        b = d[i:]
                    images = torch.autograd.Variable(xbuf[t_][b], requires_grad=False)
                    targets = torch.autograd.Variable(ybuf[t_][b], requires_grad=False)
                    task = torch.autograd.Variable(torch.LongTensor([t_]), requires_grad=False).cuda()
                    s = (self.smax - 1 / self.smax) * i / len(d) + 1 / self.smax
                    outs, _ = self.model.forward(task, images, s=s)
                    out = outs[t_]
                    loss_ = self.ce(out, targets)

                    op.zero_grad()
                    loss_.backward()
                    i = 0
                    for n, p in self.model.named_parameters():
                        if p.grad is not None:
                            if n.startswith('e'):

                                p.grad.data *= ((1 - self.old_mask[t_][i]) * self.mask_pre[i]).round().expand_as(p)

                                num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                                den = torch.cosh(p.data) + 1
                                p.grad.data *= self.smax / s * num / den
                                i += 1
                            elif n.startswith('l'):
                                if float(n[5:]) / 2 == t_:
                                    m = torch.unsqueeze(((1 - self.old_mask[t_][-1]) * self.mask_pre[-1]).round(),
                                                        0).expand_as(p)
                                    p.grad.data *= m

                    op.step()
                    for n, p in self.model.named_parameters():
                        if n.startswith('e'):
                            p.data = torch.clamp(p.data, -thres_emb, thres_emb)

                v_acc, _ = self.eval_(t_, xbuf[t_], ybuf[t_])
                if v_acc < v_best:
                    v_best = v_acc
                    self.best_model = utils.get_model(self.model, 'best')
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= self.lr_factor
                        if lr < self.lr_min:
                            break
                        for n, p in self.model.named_parameters():
                            if n.startswith('e') or n.startswith('l'):
                                p.requires_grad = True
                            else:
                                p.requires_grad = False
                        op = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
                        patience = 3

            self.model = utils.set_model_(0, self.best_model)

            if v_best >= best_acc[t_]:
                print('-' * 5, v_best, '>=', best_acc[t_], ',step back')
                # d = dict()s
                count = 0
                # print(inc)
                for n, p in self.model.named_parameters():
                    if n.startswith('e'):
                        p.data[t_] = p.data[t_] * (1 - (
                                (1 - self.old_mask[t_][count]) * self.mask_pre[count])).round() + torch.full_like(
                            p.data[t_],
                            -6).cuda() * ((1 - self.old_mask[t_][count]) * self.mask_pre[count]).round()
                        count += 1
                    elif n.startswith('l'):
                        if float(n[5:]) / 2 == t_:  # last.n/n+1判断任务id
                            m = torch.unsqueeze(((1 - self.old_mask[t_][-1]) * self.mask_pre[-1]).round(), 0).expand_as(
                                p)
                            p.data = p.data * (1 - m)
                        elif float(n[5:]) / 2 == t_ + 0.5:  # last.b
                            p.data = b_back
                # print(self.model.efc1[0], self.model.efc2[0])
                self.best_model = utils.get_model(self.model, 'bm')
                break
            else:
                print('update')
                best_acc[t_] = v_best
                self.model = utils.set_model_(0, self.best_model)
                # best= utils.get_model(self.model, 'best')

        for n, p in self.model.named_parameters():
            p.requires_grad = True
        return best_acc

    def eval(self, t, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        total_reg = 0

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = torch.autograd.Variable(x[b], volatile=True)
            targets = torch.autograd.Variable(y[b], volatile=True)
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=True)

            outputs, masks = self.model.forward(task, images, s=self.smax)
            output = outputs[t]
            loss, reg = self.criterion(output, targets, masks)
            _, pred = output.max(1)
            hits = (pred == targets).float()

            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += len(b)
            total_reg += reg.data.cpu().numpy().item() * len(b)

        return total_reg / total_num, total_loss / total_num, total_acc / total_num

    def eval_(self, t, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = torch.autograd.Variable(x[b], volatile=True)
            targets = torch.autograd.Variable(y[b], volatile=True)
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=True)

            # Forward
            outputs, masks = self.model.forward(task, images, s=self.smax)
            output = outputs[t]
            loss = self.ce(output, targets)
            _, pred = output.max(1)
            hits = (pred == targets).float()

            # Log
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += len(b)

        return total_loss / total_num, total_acc / total_num

    def criterion(self, outputs, targets, masks):
        reg = 0
        count = 0
        if self.mask_pre is not None:
            for m, mp in zip(masks, self.mask_pre):
                aux = 1 - mp
                reg += (m * aux).sum()
                count += aux.sum()
        else:
            for m in masks:
                reg += m.sum()
                count += np.prod(m.size()).item()
        reg /= count

        return self.ce(outputs, targets) + self.lamb * reg, reg

    def js_div(self, p_logits, q_logits, get_softmax=True):

        KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')
        if get_softmax:
            p_output = torch.nn.functional.softmax(p_logits)
            q_output = torch.nn.functional.softmax(q_logits)
        log_mean_output = ((p_output + q_output) / 2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

########################################################################################################################
