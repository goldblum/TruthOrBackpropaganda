"""Construct examples for local optimality and verify by eigenvalue analysis.

Check all variants
"""
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import csv
import os

import dl_myths as dl
from pytorch_cifar.models import ResNet, BasicBlock, ResNetLinear, BasicBlockLinear

from collections import defaultdict, OrderedDict
import datetime

torch.backends.cudnn.benchmark = True

"""
config:
-batch_size
-model
-device
-dtype
-epochs
-print_loss
-lr
-weight_decay
-full_batch
-init
-perturbation

"""

parser = argparse.ArgumentParser(description='Run a single job for the local optima experiment')
parser.add_argument('--net', default='MLP', type=str)
parser.add_argument('--dec', default=0.0005, type=float)
parser.add_argument('--init', default='default', type=str)
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--nosave', action='store_false')
parser.add_argument('--var', default=0.1, type=float)
parser.add_argument('--table_path', default='tables/', type=str)
parser.add_argument('--mom', default=0.4, type=float)
parser.add_argument('--GD', action='store_true')
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--bs', default=128, type=float)
args = parser.parse_args()


def run_job():
    """Wrap around experiment configs."""
    default_config = dict(
        batch_size=args.bs,
        model='ResNet',
        setup=dict(device=torch.device('cuda:0'), dtype=torch.float),
        epochs=450,
        print_loss=50,
        lr=args.lr,
        weight_decay=1e-4,
        full_batch=args.GD,
        init='default',
        switch_to_gd=10_000,
        stop_batchnorm=400,
        power_iterations=500,
        bias_variance=args.var
    )

    if args.dryrun:
        default_config['power_iterations'] = 1

    default_config['model'] = args.net
    default_config['weight_decay'] = args.dec
    default_config['init'] = args.init

    print('Currently evaluating -------------------------------:')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(default_config)
    print(args)
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.torch.cuda.device_count()}')
    # try:
    evaluation_procedure(default_config)
    # except (KeyboardInterrupt, SystemExit):
    #     raise
    # except Exception as e:
    #    print(repr(e))
    print('-----------------------------------------------------')
    print('Job finished. ---------------------------------------')
    print('-----------------------------------------------------')


def evaluation_procedure(config):
    """Train model and evaluate eigenvalues with given configuration."""
    # Setup data
    augmentations = False
    trainloader, testloader = dl.get_loaders('CIFAR10', config['batch_size'], augmentations=augmentations)

    # Setup Network
    if config['model'] == 'MLP':
        net = torch.nn.Sequential(OrderedDict([
                                 ('flatten', torch.nn.Flatten()),
                                 ('linear0', torch.nn.Linear(3072, 2048)),
                                 ('relu0', torch.nn.ReLU()),
                                 ('linear1', torch.nn.Linear(2048, 2048)),
                                 ('relu1', torch.nn.ReLU()),
                                 ('linear2', torch.nn.Linear(2048, 1024)),
                                 ('relu2', torch.nn.ReLU()),
                                 ('linear3', torch.nn.Linear(1024, 10))]))
    elif config['model'] == 'MLPsmall':
        net = torch.nn.Sequential(OrderedDict([
                                 ('flatten', torch.nn.Flatten()),
                                 ('linear0', torch.nn.Linear(3072, 256)),
                                 ('relu0', torch.nn.ReLU()),
                                 ('linear1', torch.nn.Linear(256, 256)),
                                 ('relu1', torch.nn.ReLU()),
                                 ('linear2', torch.nn.Linear(256, 256)),
                                 ('relu2', torch.nn.ReLU()),
                                 ('linear3', torch.nn.Linear(256, 10))]))
    elif config['model'] == 'MLPsmallB':
        net = torch.nn.Sequential(OrderedDict([
                                 ('flatten', torch.nn.Flatten()),
                                 ('linear0', torch.nn.Linear(3072, 256)),
                                 ('relu0', torch.nn.ReLU()),
                                 ('bn0', torch.nn.BatchNorm2d(256)),
                                 ('linear1', torch.nn.Linear(256, 256)),
                                 ('relu1', torch.nn.ReLU()),
                                 ('bn0', torch.nn.BatchNorm2d(256)),
                                 ('linear2', torch.nn.Linear(256, 256)),
                                 ('relu2', torch.nn.ReLU()),
                                 ('bn0', torch.nn.BatchNorm2d(256)),
                                 ('linear3', torch.nn.Linear(256, 10))]))
    elif config['model'] == 'ResNet':
        net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    elif config['model'] == 'L-MLP':
        net = torch.nn.Sequential(OrderedDict([
                                 ('flatten', torch.nn.Flatten()),
                                 ('linear0', torch.nn.Linear(3072, 2048)),
                                 ('linear1', torch.nn.Linear(2048, 2048)),
                                 ('linear2', torch.nn.Linear(2048, 1024)),
                                 ('linear3', torch.nn.Linear(1024, 10))]))
    elif config['model'] == 'L-ResNet':
        net = ResNetLinear(BasicBlockLinear, [2, 2, 2, 2], num_classes=10)

    net.to(**config['setup'])
    net = torch.nn.DataParallel(net)
    net.eval()

    def initialize_net(net, init):
        for name, param in net.named_parameters():
            with torch.no_grad():
                if init == 'default':
                    pass
                elif init == 'zero':
                    param.zero_()
                elif init == 'low_bias':
                    if 'bias' in name:
                        param -= 20
                elif init == 'high_bias':
                    if 'bias' in name:
                        param += 20
                elif init == 'equal':
                    torch.nn.init.constant_(param, 0.001)
                elif init == 'variant_bias':
                    if 'bias' in name:
                        torch.nn.init.uniform_(param, -args.var, args.var)
    initialize_net(net, config['init'])

    # Optimizer and loss
    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'],
                                momentum=args.mom, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Analyze model before training
    analyze_model(net, trainloader, testloader, loss_fn, config)

    # Train
    print('Starting training ...')
    dl.train(net, optimizer, scheduler, loss_fn, trainloader, config, dryrun=args.dryrun)

    # Analyze results
    acc_train, acc_test, loss_train, loss_trainw, grd_train, maxeig, mineig = analyze_model(
        net, trainloader, testloader, loss_fn, config)

    save_output(args.table_path, init=config['init'], var=args.var, acc_train=acc_train, acc_test=acc_test,
                loss_train=loss_train, loss_trainw=loss_trainw, grd_train=grd_train, maxeig=maxeig, mineig=mineig)


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
    #                                        tol=0, max_iter=config['power_iterations'])
    maxeig, mineig = (0, 0)
    print('EVs computed ...')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    # Check throughput
    if not args.dryrun:
        print('Checking throughput: ...')
        throughput_dict = dl.check_throughputs(model, trainloader, quiet=True, device=torch.device('cpu'))
        print(throughput_dict)
    return acc_train, acc_test, loss_train, loss_trainw, grd_train, maxeig, mineig


def save_output(out_dir, **kwargs):
    """Save keys to .csv files. Function from Micah."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f'table_opt_{args.net}.csv')
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


# ### ACTUAL CODE
if __name__ == "__main__":
    run_job()
