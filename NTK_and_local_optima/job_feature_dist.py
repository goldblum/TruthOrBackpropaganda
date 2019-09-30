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
parser.add_argument('--epochs', default=200, type=int, help='number of epochs for training')  # CHANGE TO 150
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

    num_params = sum([p.numel() for p in net.parameters()])
    print(f'Number of params: {num_params} - number of data points: {len(trainloader.dataset)} '
          f'- ratio : {len(trainloader.dataset) / num_params * 100:.2f}%')

    def batch_feature_correlations(dataloader, device=torch.device('cpu')):
        net.eval()
        net.to(device)
        dist_maps = list()
        cosine_maps = list()
        prod_maps = list()
        hooks = []

        def batch_wise_feature_correlation(self, input, output):
            feat_vec = input[0].detach().view(dataloader.batch_size, -1)
            dist_maps.append(torch.cdist(feat_vec, feat_vec, 2).detach().cpu().numpy())

            cosine_map = np.empty((dataloader.batch_size, dataloader.batch_size))
            prod_map = np.empty((dataloader.batch_size, dataloader.batch_size))
            for row in range(dataloader.batch_size):
                cosine_map[row, :] = torch.nn.functional.cosine_similarity(feat_vec[row:row + 1, :], feat_vec,
                                                                           dim=1, eps=1e-8).detach().cpu().numpy()
                prod_map[row, :] = torch.mean(feat_vec[row:row + 1, :] * feat_vec, dim=1).detach().cpu().numpy()
            cosine_maps.append(cosine_map)
            prod_maps.append(prod_map)

        if isinstance(net, torch.nn.DataParallel):
            hooks.append(net.module.linear.register_forward_hook(batch_wise_feature_correlation))
        else:
            if args.net in ['MLP', 'TwoLP']:
                hooks.append(net.linear3.register_forward_hook(batch_wise_feature_correlation))
            elif args.net in ['VGG', 'MobileNetV2']:
                hooks.append(net.classifier.register_forward_hook(batch_wise_feature_correlation))
            else:
                hooks.append(net.linear.register_forward_hook(batch_wise_feature_correlation))

        for inputs, _ in dataloader:
            outputs = net(inputs.to(device))
            if args.dryrun:
                break

        for hook in hooks:
            hook.remove()

        return dist_maps, cosine_maps, prod_maps

    pdist_init, cos_init, prod_init = batch_feature_correlations(trainloader)
    pdist_init_norm = np.mean([np.linalg.norm(cm.flatten()) for cm in pdist_init])
    cos_init_norm = np.mean([np.linalg.norm(cm.flatten()) for cm in cos_init])
    prod_init_norm = np.mean([np.linalg.norm(cm.flatten()) for cm in prod_init])
    print(f'The total norm of feature distances before training is {pdist_init_norm:.2f}')
    print(f'The total norm of feature cosine similarity before training is {cos_init_norm:.2f}')
    print(f'The total norm of feature inner product before training is {prod_init_norm:.2f}')

    save_plot(pdist_init, trainloader, name='pdist_before_training')
    save_plot(cos_init, trainloader, name='cosine_before_training')
    save_plot(prod_init, trainloader, name='prod_before_training')

    # Start training
    net.to(**config['setup'])
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    try:
        net.load_state_dict(torch.load(config['path'] + 'Cifar10_' + args.net + str(config["width"]) + '.pth',
                                       map_location=device))
        print('Net loaded from file.')
    except Exception as e:  # :>
        path = config['path'] + 'Cifar10_' + args.net + str(config["width"]) + '.pth'
        dl.train(net, optimizer, scheduler, loss_fn, trainloader, config, path=None, dryrun=args.dryrun)
        if not args.dryrun:
            torch.save(net.state_dict(), path)
            print('Net saved to file.')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    # Check feature maps after training
    pdist_after, cos_after, prod_after = batch_feature_correlations(trainloader)

    pdist_after_norm = np.mean([np.linalg.norm(cm.flatten()) for cm in pdist_after])
    cos_after_norm = np.mean([np.linalg.norm(cm.flatten()) for cm in cos_after])
    prod_after_norm = np.mean([np.linalg.norm(cm.flatten()) for cm in prod_after])
    print(f'The total norm of feature distances after training is {pdist_after_norm:.2f}')
    print(f'The total norm of feature cosine similarity after training is {cos_after_norm:.2f}')
    print(f'The total norm of feature inner product after training is {prod_after_norm:.2f}')

    save_plot(pdist_after, trainloader, name='pdist_after_training')
    save_plot(cos_after, trainloader, name='cosine_after_training')
    save_plot(prod_after, trainloader, name='prod_after_training')

    # Check feature map differences
    pdist_ndiff = [np.abs(co1 - co2) / pdist_init_norm for co1, co2 in zip(pdist_init, pdist_after)]
    cos_ndiff = [np.abs(co1 - co2) / cos_init_norm for co1, co2 in zip(cos_init, cos_after)]
    prod_ndiff = [np.abs(co1 - co2) / prod_init_norm for co1, co2 in zip(prod_init, prod_after)]

    pdist_ndiff_norm = np.mean([np.linalg.norm(cm.flatten()) for cm in pdist_ndiff])
    cos_ndiff_norm = np.mean([np.linalg.norm(cm.flatten()) for cm in cos_ndiff])
    prod_ndiff_norm = np.mean([np.linalg.norm(cm.flatten()) for cm in prod_ndiff])
    print(f'The total norm normalized diff of feature distances after training is {pdist_ndiff_norm:.2f}')
    print(f'The total norm normalized diff of feature cosine similarity after training is {cos_ndiff_norm:.2f}')
    print(f'The total norm normalized diff of feature inner product after training is {prod_ndiff_norm:.2f}')

    save_plot(pdist_ndiff, trainloader, name='pdist_ndiff')
    save_plot(cos_ndiff, trainloader, name='cosine_ndiff')
    save_plot(prod_ndiff , trainloader, name='prod_ndiff')

    # Check feature map differences
    pdist_rdiff = [np.abs(co1 - co2) / (np.abs(co1) + 1e-6) for co1, co2 in zip(pdist_init, pdist_after)]
    cos_rdiff = [np.abs(co1 - co2) / (np.abs(co1) + 1e-6) for co1, co2 in zip(cos_init, cos_after)]
    prod_rdiff = [np.abs(co1 - co2) / (np.abs(co1) + 1e-6) for co1, co2 in zip(prod_init, prod_after)]

    pdist_rdiff_norm = np.mean([np.linalg.norm(cm.flatten()) for cm in pdist_rdiff])
    cos_rdiff_norm = np.mean([np.linalg.norm(cm.flatten()) for cm in cos_rdiff])
    prod_rdiff_norm = np.mean([np.linalg.norm(cm.flatten()) for cm in prod_rdiff])
    print(f'The total norm relative diff of feature distances after training is {pdist_rdiff_norm:.2f}')
    print(f'The total norm relative diff of feature cosine similarity after training is {cos_rdiff_norm:.2f}')
    print(f'The total norm relative diff of feature inner product after training is {prod_rdiff_norm:.2f}')

    save_plot(pdist_rdiff, trainloader, name='pdist_rdiff')
    save_plot(cos_rdiff, trainloader, name='cosine_rdiff')
    save_plot(prod_rdiff , trainloader, name='prod_rdiff')

    save_output(args.table_path, width=config['width'], num_params=num_params,
                pdist_init_norm=pdist_init_norm, pdist_after_norm=pdist_after_norm, pdist_ndiff_norm=pdist_ndiff_norm,
                pdist_rdiff_norm=pdist_rdiff_norm,
                cos_init_norm=pdist_init_norm, cos_after_norm=pdist_after_norm, cos_ndiff_norm=pdist_ndiff_norm,
                cos_rdiff_norm=cos_rdiff_norm,
                prod_init_norm=pdist_init_norm, prod_after_norm=pdist_after_norm, prod_ndiff_norm=pdist_ndiff_norm,
                prod_rdiff_norm=prod_rdiff_norm)

    # Save raw data
    raw_pkg = dict(pdist_init=pdist_init, cos_init=cos_init, prod_init=prod_init,
                   pdist_after=pdist_after, cos_after=cos_after, prod_after=prod_after,
                   pdist_ndiff=pdist_ndiff, cos_ndiff=cos_ndiff, prod_ndiff=prod_ndiff,
                   pdist_rdiff=pdist_rdiff, cos_rdiff=cos_rdiff, prod_rdiff=prod_rdiff)
    path = config['path'] + 'Cifar10_' + args.net + str(config["width"]) + '_rawmaps.pth'
    torch.save(raw_pkg, path)

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


def save_output(out_dir, **kwargs):
    """Save keys to .csv files. Function from Micah."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f'table_{args.net}.csv')
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = [line for line in reader][0]
    except Exception as e:
        print(repr(e))
        with open(fname, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()

    # Add row for this experiment
    with open(fname, 'a') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        writer.writerow(kwargs)
    print('\nResults saved to ' + fname + '.')


if __name__ == '__main__':
    main()
