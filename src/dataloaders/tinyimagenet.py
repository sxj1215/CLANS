import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from shutil import move
from os import rmdirs
import glob
#import ossaudiodev
from shutil import copy
import random

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


def experience():
    # 获取data文件夹下所有文件夹名（即需要分类的类名）
    file_path = './tiny-imagenet-200/train'
    new_file_path ='./tiny-imagenet-200/exp'
    # 划分比例，训练集 : 验证集 = 8 : 2
    split_rate = 0.05

    data_class = [cla for cla in os.listdir(file_path)]

    train_path = new_file_path
    mkfile(new_file_path)

    for cla in data_class:
        cla_path = file_path + '/' + cla + '/' +'images' + '/'
        mkfile(cla_path)


        images = sorted(os.listdir(cla_path))
        eval_index = 100
        for index, image in enumerate(images):
            if index < eval_index:
                image_path = cla_path + '/' + image
                new_path = new_file_path + '/' + cla + '/'
                mkfile(new_path)
                copy(image_path, new_path)
            if index == 100:
                break



########################################################################################################################
def get(seed=0, fixed_order=False, pc_valid=0):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    xbuf = {}
    ybuf = {}
    nperm = 10
    seeds = np.array(list(range(nperm)), dtype=int)
    if not fixed_order:
        seeds = shuffle(seeds, random_state=seed)

    if not os.path.isdir('../dat/binary_tiny'):
        os.makedirs('../dat/binary_tiny')
        # Pre-load
        # MNIST

        data_dir = './tiny-imagenet-200'
        #val()
        experience()


        dat = {}
        xbuf = {}
        ybuf = {}
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             normalize, ])
        transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
        transform_exp = transforms.Compose(
            [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             normalize, ])
        trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        expset = datasets.ImageFolder(root=os.path.join(data_dir, 'exp'), transform=transform_train)
        testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        dat['train'] = trainset
        dat['test'] = testset
        dat['exp'] = expset
        for n in range(10):
            xbuf[n] = {}
            ybuf[n] = {}
            xbuf[n]['train'] = {'x': []}
            ybuf[n]['train'] = {'y': []}
            data[n] = {}
            data[n]['name'] = 'tiny'
            data[n]['ncla'] = 20
            data[n]['train'] = {'x': [], 'y': []}
            data[n]['test'] = {'x': [], 'y': []}
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                task_idx = target.numpy()[0] // 20
                data[task_idx][s]['x'].append(image)
                data[task_idx][s]['y'].append(target.numpy()[0] % 20)
        loader1 = torch.utils.data.DataLoader(dat['exp'], batch_size=1, shuffle=False)
        for image, target in loader1:
            task_idx = target.numpy()[0] // 20
            xbuf[task_idx]['train']['x'].append(image)
            ybuf[task_idx]['train']['y'].append(target.numpy()[0] % 20)


            # "Unify" and save
        for t in range(10):
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                if s == 'train':
                    xbuf[t][s]['x'] = torch.stack(xbuf[t][s]['x']).view(-1, size[0], size[1], size[2])
                    ybuf[t][s]['y'] = torch.LongTensor(np.array(ybuf[t][s]['y'], dtype=int)).view(-1)
                    torch.save(xbuf[t][s]['x'], os.path.join(os.path.expanduser('../dat/binary_tiny'),
                                                             'data' + str(t) + s + 'expx.bin'))
                    torch.save(ybuf[t][s]['y'], os.path.join(os.path.expanduser('../dat/binary_tiny'),
                                                             'data' + str(t) + s + 'expy.bin'))
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser('../dat/binary_tiny'),
                                                         'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser('../dat/binary_tiny'),
                                                         'data' + str(t) + s + 'y.bin'))
        print()

    else:

        # Load binary files
        for i, r in enumerate(seeds):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = 20
            data[i]['name'] = 'tiny-{:d}'.format(i)
            xbuf[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            ybuf[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            xbuf[i]['ncla'] = 20
            ybuf[i]['ncla'] = 20
            # Load
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                xbuf[i][s] = {'x': []}
                ybuf[i][s] = {'x': []}
                data[i][s]['x'] = torch.load(
                    os.path.join(os.path.expanduser('../dat/binary_tiny'), 'data' + str(r) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(
                    os.path.join(os.path.expanduser('../dat/binary_tiny'), 'data' + str(r) + s + 'y.bin'))
                if s == 'train':
                    xbuf[i][s]['x'] = torch.load(
                        os.path.join(os.path.expanduser('../dat/binary_tiny'), 'data' + str(r) + s + 'expx.bin'))
                    ybuf[i][s]['y'] =  torch.load(
                        os.path.join(os.path.expanduser('../dat/binary_tiny'), 'data' + str(r) + s + 'expy.bin'))
    # Validation
    for t in data.keys():
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size, xbuf, ybuf


def val():
    target_folder = './tiny-imagenet-200/val/'

    val_dict = {}
    with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = glob.glob('./tiny-imagenet-200/val/images/*')
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
            os.mkdir(target_folder + str(folder) + '/images')

    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        dest = target_folder + str(folder) + '/images/' + str(file)
        move(path, dest)

    rmdir('./tiny-imagenet-200/val/images')


