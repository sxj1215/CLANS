import math
import warnings
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

warnings.filterwarnings('ignore')
import random
import sys, os, argparse, time
import numpy as np
import torch
from torch import optim

torch.set_printoptions(profile='full')
import utils
import copy
from RL import rl_model as AC

tstart = time.time()

# Arguments
parser = argparse.ArgumentParser(description='xxx')
parser.add_argument('--lamb', type=int, default=0.75)
parser.add_argument('--space', type=int, default=30, help="the state space for search")
parser.add_argument('--a', type=int, default=0.25)
parser.add_argument('--b', type=int, default=1)
parser.add_argument('--c', type=int, default=1)
parser.add_argument('--d', type=int, default=0.75)
parser.add_argument('--n', type=int, default=10)
parser.add_argument('--smax', type=int, default=400)
parser.add_argument('--max_trials', type=int, default=1)
parser.add_argument('--penalty', type=float, default=0.0001)
parser.add_argument('--actions_num', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
parser.add_argument('--experiment', default='tiny', type=str, required=False,
                    choices=['tiny'], help='(default=%(default)s)')
parser.add_argument('--approach', default='rl_hat', type=str, required=False,
                    choices=['rl_hat'])
parser.add_argument('--output', default='', type=str, required=False)
parser.add_argument('--nepochs', default=80, type=int, required=False)
parser.add_argument('--lr', default=0.05, type=float, required=False)
parser.add_argument('--parameter', type=str, default='')
parser.add_argument('--eblation', type=str, default='')
args = parser.parse_args()
if args.output == '':
    args.output = '../res/' + args.experiment + '_' + args.approach + '_' + str(args.seed) + '.txt'
print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:

    print('[CUDA unavailable]')
    sys.exit()

# Args -- Experiment
if args.experiment == 'tiny':
    from dataloaders import tinyimagenet as dataloader

# Args -- Approach

if args.approach == 'rl_hat':
    from approaches import rl_hat as approach

# Args -- Network
if args.experiment == 'pmnist' or args.experiment == 'rmnist':
    if args.approach == 'rl_hat':
        from networks import mlp_rl as network

else:
    if args.approach == 'rl_hat':
        from networks import alexnet_rl as network


def js_div(p_logits, q_logits, get_softmax=True):
    KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = torch.nn.functional.softmax(p_logits.view(1, -1), dim=1)
        q_output = torch.nn.functional.softmax(q_logits.view(1, -1), dim=1)
    log_mean_output = ((p_output + q_output) / 2).log()
    return 1 - (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


def get_masks(net, t):
    old_mask = []
    for i in range(t + 1):
        masks = net.mask(i, s=args.smax)
        masks_ = []
        for m in masks:
            masks_.append(m.detach().clone())
        for n in range(len(masks_)):
            masks_[n] = torch.round(masks_[n])
        masks = masks_
        if i == 0:
            prev = masks
        else:
            prev_ = []
            for m1, m2 in zip(prev, masks):
                prev_.append(torch.max(m1, m2))
            prev = prev_
        old_mask.append(masks)
    old_maskpre = prev
    return old_mask, old_maskpre


def calculate_complexity(masks, mask_pre):
    count = 0
    for m, mp in zip(masks, mask_pre):
        old_size = mp.shape
        new_size = m.shape
        try:
            count += list(new_size)[0] / list(old_size)[1]
        except IndexError:
            count += list(new_size)[0] / list(old_size)[0]

    return count / len(masks)


def random_unit(p: float):
    if p == 0:
        return False
    if p == 1:
        return True

    R = random.random()
    if R < p:
        return True
    else:
        return False


########################################################################################################################
average = []
average2 = []
average3 = []
total = [[[[]]]]
for i in range(1):
    # Load
    average1 = []
    firstAccuracy = []
    bwt = []
    print('Load data...')
    data, taskcla, inputsize, xbuf1, ybuf1 = dataloader.get(seed=args.seed)
    print('Input size =', inputsize, '\nTask info =', taskcla)

    # Inits
    print('Inits...')
    if args.approach == 'rl_hat' or args.approach == 'rcl':
        if args.experiment == 'pmnist' or args.experiment == 'rmnist':
            action_num = 2
        else:
            action_num = 3
        net = network.Net(dict(), dict(), None, None, None, inputsize, taskcla).cuda()
        appr = approach.Appr(net, None, None, args.nepochs, lr=args.lr, args=args)

    else:
        print(args.approach)
        net = network.Net(inputsize, taskcla).cuda()
        appr = approach.Appr(net, nepochs=args.nepochs, lr=args.lr, args=args)
    utils.print_model_report(net)

    utils.print_optimizer_config(appr.optimizer)
    print('-' * 100)

    # Loop tasks
    acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

    xbuf = []
    ybuf = []
    old_mask = []
    old_maskpre = []
    valid_acc = []
    neuronused = torch.zeros((9, 100))
    quxian = []
    for t, ncla in taskcla:
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(t, data[t]['name']))
        print('*' * 100)

        task_pre = []
        if args.approach == 'rl_hat':
            xtrain = data[t]['train']['x'].cuda()
            ytrain = data[t]['train']['y'].cuda()
            xvalid = data[t]['valid']['x'].cuda()
            yvalid = data[t]['valid']['y'].cuda()
            task = t

            xbuf.append(xbuf1[t]['train']['x'].cuda())
            ybuf.append(ybuf1[t]['train']['y'].cuda())

        # Train
        if args.approach == 'rl_hat':
            if t > 0:
                if args.experiment == 'amnist' or args.experiment == 'pmnist' or args.experiment == 'rmnist':
                    old_W = dict()
                    old_b = dict()
                    old_embedding = dict()
                    old_last = net_temp.last
                else:
                    old_conv = dict()
                    old_fc = dict()
                    old_embedding = dict()
                    old_last = net_temp.last
                for n, p in net_temp.named_parameters():
                    if n.startswith('w'):
                        old_W[n] = p
                    elif n.startswith('b'):
                        old_b[n] = p
                    elif n.startswith('e'):
                        old_embedding[n] = p
                    elif n.startswith('c'):
                        old_conv[n] = p
                    elif n.startswith('f'):
                        old_fc[n] = p

                lr = 0.03
                betas = (0.9, 0.999)

                policy = AC.ActorCritic(t, args.space)
                best_reward = 0
                best_actions = [0] * action_num
                optimizer = torch.optim.Adam(policy.parameters(), lr=lr, betas=betas)

                standard_acc = 1

                similarity_data = []
                for t_ in range(t):
                    similarity_data.append(js_div(xbuf[t_], xbuf[t]))

                for i, m in enumerate(old_mask):
                    print('task', i, m[0].sum(), m[1].sum(),
                          m[2].sum())
                for i_episode in range(args.max_trials):
                    with torch.no_grad():
                        temp_pre = copy.deepcopy(old_maskpre)
                        temp_masks = copy.deepcopy(old_mask)

                    state = similarity_data
                    if random_unit(0.3):
                        state = torch.rand(1, t)
                    if i_episode < args.max_trials:
                        print('trial ', i_episode + 1, '*' * 91)
                        actions = policy(torch.as_tensor(state), action_num)
                    else:
                        print('rl test', '*' * 92)
                        actions = best_actions

                    for i, m in enumerate(temp_pre):
                        if i < action_num:
                            try:
                                temp_pre[i] = torch.cat((temp_pre[i], torch.zeros(1, actions[i]).cuda()), dim=1)
                            except IndexError:
                                temp_pre[i] = torch.cat(
                                    (torch.unsqueeze(temp_pre[i], 0), torch.zeros(1, actions[i]).cuda()), dim=1)
                    for t_ in range(t):
                        task_pre.append(t_)
                        for i, m in enumerate(temp_pre):
                            if i < action_num:

                                try:
                                    temp_masks[t_][i] = torch.cat(
                                        (temp_masks[t_][i], torch.zeros(1, actions[i]).cuda()), dim=1)
                                except IndexError:
                                    temp_masks[t_][i] = torch.cat(
                                        (torch.unsqueeze(temp_masks[t_][i], 0), torch.zeros(1, actions[i]).cuda()),
                                        dim=1)

                    if args.experiment == 'amnist' or args.experiment == 'pmnist' or args.experiment == 'rmnist':
                        net = network.Net(old_W, old_b, old_last, old_embedding, actions, inputsize, taskcla,
                                          t=t).cuda()
                    else:
                        net = network.Net(old_conv, old_fc, old_last, old_embedding, actions, inputsize, taskcla,
                                          t=t).cuda()

                    appr = approach.Appr(net, temp_pre, temp_masks, args.nepochs, lr=args.lr, args=args)
                    appr.create_mask_back()
                    print(' actions:', [actions[i] for i in range(action_num)])
                    appr.train(t, xtrain, ytrain, xvalid, yvalid)

                    similarity = []
                    attention = []
                    for i in range(t + 1):
                        mask = net.mask(i, s=args.smax)
                        attention.append(mask)
                    for m in attention:
                        cos = 0
                        # s2=0
                        for i in range(action_num):
                            cos += torch.cosine_similarity(m[i], attention[t][i], dim=0)

                        similarity.append(cos / action_num)

                    complexity = calculate_complexity(mask, old_maskpre)
                    _, ac = appr.eval_(t, xvalid, yvalid)
                    state = similarity[:-1]

                    reward = ac / standard_acc
                    reward = float(30 * actions[0] - 30 * actions[1])
                    if reward > best_reward:
                        best_reward = reward
                        best_actions = actions

                    print('reward', reward, 'new_acc', ac, 'comp', complexity)
                    # new_acc = ac
                    policy.rewards.append(reward)

                    if i_episode < args.max_trials:
                        print('training RL agent...')
                        optimizer.zero_grad()
                        loss = policy.calculateLoss(action_num)
                        loss.backward()
                        optimizer.step()
                        policy.clearMemory()

                # 知识转移

                print('transferring knowledge...')
                print('before:', valid_acc)

                old_mask, old_maskpre = get_masks(appr.model, t)
                mask_print = torch.zeros((9, 100))
                for i, m in enumerate(old_mask):
                    print('task', i, m[0].sum(), m[1].sum(),
                          m[2].sum())  # ,m[0], m[1])#,m[2].sum(),m[3].sum(),m[4].sum())

                appr.mask_pre = old_maskpre
                appr.old_mask = old_mask
                task_pre = task_pre[:-1]
                taskWithSimilarity = zip(task_pre, state)
                valid_acc = appr.transfer(t, xbuf, ybuf, args.n, valid_acc, taskWithSimilarity)
                print('after:', valid_acc)
                net_temp = utils.set_model_(0, appr.best_model)

                old_mask, old_maskpre = get_masks(net_temp, t)
                for i, m in enumerate(old_mask):
                    print('task', i, m[0].sum(), m[1].sum())

                v_acc, _ = appr.eval_(t, xvalid, yvalid)
                valid_acc.append(v_acc)
                fig = plt.figure()
                a = np.array(mask_print.cpu())

                print('total', old_maskpre[0].sum(), old_maskpre[1].sum())  # ,old_maskpre[0], old_maskpre[1])

            else:
                appr.train(t, xtrain, ytrain, xvalid, yvalid)
                torch.save(appr.model, '.\parameter.pkl')
                net_temp = torch.load('.\parameter.pkl')
                old_maskpre = net_temp.mask(t, s=args.smax)
                a = []
                for m in old_maskpre:
                    a.append(m.detach().clone())
                for i in range(len(a)):
                    a[i] = torch.round(a[i])
                old_maskpre = a
                old_mask.append(a)

                v_acc, _ = appr.eval_(t, xvalid, yvalid)
                valid_acc.append(v_acc)
                print(a[0].sum(), a[1].sum())

        print('-' * 100)

        # Test
        for u in range(t + 1):
            xtest = data[u]['test']['x'].cuda()
            ytest = data[u]['test']['y'].cuda()
            if args.approach == 'rl_hat':
                test_loss, test_acc = appr.eval_(u, xtest, ytest)
            else:
                test_loss, test_acc = appr.eval(u, xtest, ytest)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                          100 * test_acc))
            acc[t, u] = test_acc
            lss[t, u] = test_loss

        # Save
        if args.approach == 'rl_hat':
            print('layer1:', appr.model.c1b.shape[0], 'layer2:', appr.model.c2b.shape[0], 'layer3:',
                  appr.model.c3b.shape[0])
        print('Save at ' + args.output)

    print('*' * 100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t', end='')
        for j in range(acc.shape[1]):
            print('{:5.2f}% '.format(100 * acc[i, j]), end='')
        print()
    print('*' * 100)
    print('Done!')

    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

    if hasattr(appr, 'logs'):
        if appr.logs is not None:
            # save task names
            from copy import deepcopy

            appr.logs['task_name'] = {}
            appr.logs['test_acc'] = {}
            appr.logs['test_loss'] = {}
            for t, ncla in taskcla:
                appr.logs['task_name'][t] = deepcopy(data[t]['name'])
                appr.logs['test_acc'][t] = deepcopy(acc[t, :])
                appr.logs['test_loss'][t] = deepcopy(lss[t, :])
            # pickle
            import gzip
            import pickle

            with gzip.open(os.path.join(appr.logpath), 'wb') as output:
                pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
