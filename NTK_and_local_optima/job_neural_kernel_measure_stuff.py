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
    param_norm_before = np.sqrt(np.sum([p.pow(2).sum().detach().cpu().numpy() for p in net.parameters()]))
    print(f'The L2 norm of the parameter vector is {param_norm_before:.2f}')

    net_init = [p.detach().clone() for p in net.parameters()]

    # Start training
    net.to(**config['setup'])
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    loss_fn = torch.nn.CrossEntropyLoss()
    analyze_model(net, trainloader, testloader, loss_fn, config)
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    try:
        net.load_state_dict(torch.load(config['path'] + 'Cifar10_' + args.net + str(config["width"]) + '_after.pth',
                                       map_location=device))
        print('Net loaded from file.')
    except Exception as e:  # :>
        print(repr(e))
        print('Could not find model data ... aborting ...')
        return
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    param_norm_after = np.sqrt(np.sum([p.pow(2).sum().detach().cpu().numpy() for p in net.parameters()]))
    print(f'The L2 norm of the parameter vector is {param_norm_after:.2f}')

    change_total = 0.0
    for p1, p2 in zip(net_init, net.parameters()):
        change_total += (p1 - p2).detach().pow(2).sum()
    change_total = change_total.sqrt().cpu().numpy()

    change_rel = 0.0
    for p1, p2 in zip(net_init, net.parameters()):
        change_rel += (p1 - p2).detach().pow(2).mean()
    change_rel = change_rel.sqrt().cpu().numpy()

    change_nrmsum = 0.0
    for p1, p2 in zip(net_init, net.parameters()):
        change_nrmsum += (p1 - p2).norm()
    change_nrmsum = change_nrmsum.cpu().numpy()

    # Analyze results
    acc_train, acc_test, loss_train, loss_trainw, grd_train = analyze_model(
        net, trainloader, testloader, loss_fn, config)

    save_output(args.table_path, name='ntk_stats', width=config['width'], num_params=num_params,
                acc_train=acc_train, acc_test=acc_test, loss_train=loss_train, loss_trainw=loss_trainw,
                grd_train=grd_train, param_norm_before=param_norm_before, param_norm_after=param_norm_after,
                change_total=change_total, change_rel=change_rel, change_nrmsum=change_nrmsum)

    # Save raw data
    # raw_pkg = dict(pdist_init=pdist_init, cos_init=cos_init, prod_init=prod_init,
    #                pdist_after=pdist_after, cos_after=cos_after, prod_after=prod_after,
    #                pdist_ndiff=pdist_ndiff, cos_ndiff=cos_ndiff, prod_ndiff=prod_ndiff,
    #                pdist_rdiff=pdist_rdiff, cos_rdiff=cos_rdiff, prod_rdiff=prod_rdiff)
    # path = config['path'] + 'Cifar10_' + args.net + str(config["width"]) + '_rawmaps.pth'
    # torch.save(raw_pkg, path)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('-----------------------------------------------------')
    print('Job finished.----------------------------------------')
    print('-----------------------------------------------------')


def analyze_model(model, trainloader, testloader, loss_fn, config):
    """Get accuracy, loss, 1st order optimality and 2nd order optimality for model."""
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    model.to(**config['setup'])
    # Accuracy and loss
    acc_train = (100 * dl.get_accuracy(model, trainloader, config))
    acc_test = (100 * dl.get_accuracy(model, testloader, config))
    print(f'Accuracy of the network on training images: {acc_train} %%')
    print(f'Accuracy of the network on test images: {acc_test} %%')
    loss_train = dl.compute_loss(model, loss_fn, trainloader, config, add_weight_decay=False)
    print(f'Loss in training is {loss_train:.12f}')
    loss_trainw = dl.compute_loss(model, loss_fn, trainloader, config, add_weight_decay=True)
    print(f'Loss in training (+L2 Reg) is {loss_trainw:.12f}')
    print(f'Loss in testing is {dl.compute_loss(model, loss_fn, testloader, config, add_weight_decay=False):.12f}')
    # 1st order opt
    grd_train = dl.gradient_norm(trainloader, model, loss_fn, config['setup']['device'], config['weight_decay']).item()
    grd_test = dl.gradient_norm(testloader, model, loss_fn, config['setup']['device'], config['weight_decay']).item()
    print(f'Gradient norm in training is {grd_train:.12f}')
    print(f'Gradient norm in testing is {grd_test:.12f}')
    # 2nd order opt
    # hessian = dl.HessianOperator(model, trainloader, loss_fn, weight_decay=config['weight_decay'], **config['setup'])
    # maxeig, mineig = dl.eigenvalue_analysis(hessian, method='power_method',
    #                                         tol=0, max_iter=config['power_iterations'])
    # print('EVs computed ...')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    # Check throughput
    # if not args.dryrun:
    #     print('Checking throughput: ...')
    #     throughput_dict = dl.check_throughputs(model, trainloader, quiet=True, device=torch.device('cpu'))
    #     print(throughput_dict)
    return acc_train, acc_test, loss_train, loss_trainw, grd_train


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


def batch_wise_ntk(net, dataloader, device=torch.device('cpu'), samplesize=10):
    r"""Evaluate NTK on a batch sample level.

    1) Draw a batch of images from the batch
    2) Compute gradients w.r.t to all logits for all images
    3) compute n_logits² matrix by pairwise multiplication of all grads and summing over parameters
    4) Tesselate batch_size² matrix with n_logits²-sized submatrices

    1) Choose 10 images
    2) For each image pair, compute \nabla_theta F(x, theta) and \nabla_theta F(y, theta), both in R^{p x N_logits}
       then take the product of these quantities to get an N_logitsxN_logits matrix.
       This matrix will be 10x10 since you have 10 logits.
    """
    net.eval()
    net.to(device)
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device=device), targets.to(device=device)
        # grad_outputs should be a sequence of length matching output containing the “vector” in
        # Jacobian-vector product, usually the pre-computed gradients w.r.t. each of the outputs.
        # If an output doesn’t require_grad, then the gradient can be None).

        # Sort sample
        targets, indices = torch.sort(targets)
        inputs = inputs[indices, :]
        logit_dim = 10

        ntk_sample = []

        for ni in range(samplesize):                # image 1
            ntk_row = []
            for nj in range(samplesize):            # image 2
                ntk_entry = np.empty((logit_dim, logit_dim))
                for i in range(logit_dim):          # iterate over logits
                    for j in range(logit_dim):
                        prod = 0
                        net.zero_grad()
                        imgrad_ni_i = torch.autograd.grad(
                            net(inputs[ni:ni + 1, :, :, :]).squeeze()[i], net.parameters(),
                            only_inputs=True, retain_graph=True)
                        imgrad_nj_j = torch.autograd.grad(
                            net(inputs[nj:nj + 1, :, :, :]).squeeze()[j], net.parameters(),
                            only_inputs=True, retain_graph=True)
                        for p1, p2 in zip(imgrad_ni_i, imgrad_nj_j):
                            outer = (p1 * p2).sum().cpu().numpy()
                            if np.isfinite(outer):
                                prod += outer
                        ntk_entry[i, j] = prod
                        # print(f'Computed Outer product {prod} for logits {i,j}')
                ntk_row.append(ntk_entry)
                # print(f'Images ({ni},{nj}) processed.')
            ntk_sample.append(ntk_row)

        # Retile to matrix
        ntk_matrix = np.block(ntk_sample)
        return ntk_matrix


def batch_feature_correlations(dataloader, device=torch.device('cpu')):
    """Feature Corr."""
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


if __name__ == '__main__':
    main()
