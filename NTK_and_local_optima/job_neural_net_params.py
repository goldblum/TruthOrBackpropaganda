"""Analyze NTKs."""
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy
import datetime
from collections import OrderedDict
import os
import csv

import matplotlib.pyplot as plt

import dl_myths as dl
from pytorch_cifar.models import WideResNet, BasicBlock, ResNet18
from WideResNet_pytorch.networks import Wide_ResNet
from torchvision.models import MobileNetV2, VGG
from torchvision.models.vgg import make_layers

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Analyze ntks')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epochs', default=600, type=int, help='number of epochs for training')  # CHANGE TO 150
parser.add_argument('--switch_to_gd', default=10_000, type=int)
parser.add_argument('--stop_batchnorm', default=10_000, type=int)
parser.add_argument('--full_batch', action='store_true')
parser.add_argument('--path', default='/cmlscratch/jonas0/DL_myth_data/', type=str)
parser.add_argument('--table_path', default='tables/', type=str)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--print', default=50, type=int)
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--net', default='MLP', type=str)
parser.add_argument('--rerun', action='store_true')
parser.add_argument('--pdist', action='store_true')
parser.add_argument('--sampling', default=25, type=int)

# debug
parser.add_argument('--dryrun', action='store_true')

args = parser.parse_args()

if args.net != 'MobileNetV2':
    args.width = int(args.width)

config = dict()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float

config['setup'] = dict(device=device, dtype=dtype)
config['batch_size'] = args.bs
config['epochs'] = args.epochs
config['print_loss'] = args.print
config['weight_decay'] = args.weight_decay

config['lr'] = args.lr
config['switch_to_gd'] = args.switch_to_gd
config['stop_batchnorm'] = args.stop_batchnorm
config['full_batch'] = args.full_batch
config['path'] = args.path
config['width'] = args.width

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    """Check ntks in a single call."""
    print(f'RUNNING NTK EXPERIMENT WITH NET {args.net} and WIDTH {args.width}')
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.torch.cuda.device_count()}')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))

    trainloader, testloader = dl.get_loaders('CIFAR10', config['batch_size'], augmentations=False, shuffle=False)

    if args.net == 'ResNet':
        net = WideResNet(BasicBlock, [2, 2, 2, 2], widen_factor=config['width'])
    elif args.net == 'WideResNet':  # meliketoy wideresnet variant
        net = Wide_ResNet(depth=16, widen_factor=config['width'], dropout_rate=0.0, num_classes=10)
    elif args.net == 'MLP':
        net = torch.nn.Sequential(OrderedDict([
                                 ('flatten', torch.nn.Flatten()),
                                 ('linear0', torch.nn.Linear(3072, config['width'])),
                                 ('relu0', torch.nn.ReLU()),
                                 ('linear1', torch.nn.Linear(config['width'], config['width'])),
                                 ('relu1', torch.nn.ReLU()),
                                 ('linear2', torch.nn.Linear(config['width'], config['width'])),
                                 ('relu2', torch.nn.ReLU()),
                                 ('linear3', torch.nn.Linear(config['width'], 10))]))
    elif args.net == 'TwoLP':
        net = torch.nn.Sequential(OrderedDict([
                                 ('flatten', torch.nn.Flatten()),
                                 ('linear0', torch.nn.Linear(3072, config['width'])),
                                 ('relu0', torch.nn.ReLU()),
                                 ('linear3', torch.nn.Linear(config['width'], 10))]))
    elif args.net == 'MobileNetV2':
        net = MobileNetV2(num_classes=10, width_mult=config['width'], round_nearest=4)
    elif args.net == 'VGG':
        cfg_base = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        cfg = [c * config['width'] for c in cfg_base if isinstance(c, int)]
        print(cfg)
        net = VGG(make_layers(cfg), num_classes=10)
        net.classifier[0] = torch.nn.Linear(512 * 7 * 7 * config['width'], 4096)
    elif args.net == 'ConvNet':
        net = torch.nn.Sequential(OrderedDict([
                                  ('conv0', torch.nn.Conv2d(3, 1 * config['width'], kernel_size=3, padding=1)),
                                  ('relu0', torch.nn.ReLU()),
                                  # ('pool0', torch.nn.MaxPool2d(3)),
                                  ('conv1', torch.nn.Conv2d(1 * config['width'],
                                                            2 * config['width'], kernel_size=3, padding=1)),
                                  ('relu1', torch.nn.ReLU()),
                                  #  ('pool1', torch.nn.MaxPool2d(3)),
                                  ('conv2', torch.nn.Conv2d(2 * config['width'],
                                                            2 * config['width'], kernel_size=3, padding=1)),
                                  ('relu2', torch.nn.ReLU()),
                                  # ('pool2', torch.nn.MaxPool2d(3)),
                                  ('conv3', torch.nn.Conv2d(2 * config['width'],
                                                            4 * config['width'], kernel_size=3, padding=1)),
                                  ('relu3', torch.nn.ReLU()),
                                  ('pool3', torch.nn.MaxPool2d(3)),
                                  ('conv4', torch.nn.Conv2d(4 * config['width'],
                                                            4 * config['width'], kernel_size=3, padding=1)),
                                  ('relu4', torch.nn.ReLU()),
                                  ('pool4', torch.nn.MaxPool2d(3)),
                                  ('flatten', torch.nn.Flatten()),
                                  ('linear', torch.nn.Linear(36 * config['width'], 10))
                                  ]))
    else:
        raise ValueError('Invalid network specified.')
    net.to(**config['setup'])

    try:
        net.load_state_dict(torch.load(config['path'] + 'Cifar10_' + args.net + str(config["width"]) + '_before.pth',
                                       map_location=device))
        print('Initialized net loaded from file.')
    except Exception as e:  # :>
        path = config['path'] + 'Cifar10_' + args.net + str(config["width"]) + '_before.pth'
        if not args.dryrun:
            torch.save(net.state_dict(), path)
            print('Initialized net saved to file.')
        else:
            print(f'Would save to {path}')

    num_params = sum([p.numel() for p in net.parameters()])
    print(f'Number of params: {num_params} - number of data points: {len(trainloader.dataset)} '
          f'- ratio : {len(trainloader.dataset) / num_params * 100:.2f}%')

    # Start training
    net.to(**config['setup'])
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    try:
        net.load_state_dict(torch.load(config['path'] + 'Cifar10_' + args.net + str(config["width"]) + '_after.pth',
                                       map_location=device))
        print('Net loaded from file.')
    except Exception as e:  # :>
        path = config['path'] + 'Cifar10_' + args.net + str(config["width"]) + '_after.pth'
        dl.train(net, optimizer, scheduler, loss_fn, trainloader, config, path=None, dryrun=args.dryrun)
        if not args.dryrun:
            torch.save(net.state_dict(), path)
            print('Net saved to file.')
        else:
            print(f'Would save to {path}')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    save_output(args.table_path, name='ntk', width=config['width'], num_params=num_params,
                before_norm=ntk_matrix_before_norm, after_norm=ntk_matrix_after_norm,
                diff_norm=ntk_matrix_diff_norm, rdiff_norm=ntk_matrix_rdiff_norm,
                param_norm_before=param_norm_before, param_norm_after=param_norm_after,
                corr_coeff=corr_coeff, corr_tom=corr_tom)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('-----------------------------------------------------')
    print('Job finished.----------------------------------------')
    print('-----------------------------------------------------')


def save_plot(cmaps, dataloader, name='before'):
    """Save cmap to file."""
    iterable = iter(dataloader)
    _, next_targets = next(iterable)
    _, indices = torch.sort(next_targets)
    cmap = cmaps[0][indices, :][:, indices]
    plt.imshow(cmap)
    # plt.title(f'{args.net}{config["width"]} on CIFAR {name}. The total norm is {np.linalg.norm(cmap):.2f}')
    plt.savefig(config['path'] + f'{args.net}{config["width"]}_CIFAR_{name}.png', bbox_inches='tight', dpi=1200)


def save_output(out_dir, name, **kwargs):
    """Save keys to .csv files. Function from Micah."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f'table_{args.net}_{name}.csv')
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = [line for line in reader][0]
    except Exception as e:
        print('Creating a new .csv table...')
        with open(fname, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
    if not args.dryrun:
        # Add row for this experiment
        with open(fname, 'a') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writerow(kwargs)
        print('\nResults saved to ' + fname + '.')
    else:
        print(f'Would save results to {fname}.')


if __name__ == '__main__':
    main()
