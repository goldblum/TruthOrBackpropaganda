"""Construct examples for local optimality and verify by eigenvalue analysis.

Check all variants
"""
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import dl_myths as dl
from pytorch_cifar.models import ResNet, BasicBlock, ResNetLinear, BasicBlockLinear

from collections import defaultdict, OrderedDict
import datetime
import sys

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

parser = argparse.ArgumentParser(description='Run a single job for the local optima from subnets experiment')
parser.add_argument('--net', default='MLP', type=str)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--gradpen', default=0.0, type=float)
parser.add_argument('--centpen', default=0.0, type=float)
parser.add_argument('--gradual', action='store_true')
parser.add_argument('--power_its', default=500, type=int)

# debug
parser.add_argument('--debug', action='store_true')
parser.add_argument('--dryrun', action='store_true')
args = parser.parse_args()


def run_job():
    """Wrap around experiment configs."""
    default_config = dict(
        batch_size=64,
        model='MLP',
        setup=dict(device=torch.device('cuda:0'), dtype=torch.float),
        epochs_linear=100,
        epochs_distill=300,
        print_loss=10,
        lr=0.1,
        weight_decay=0,
        full_batch=False,
        switch_to_gd=10_000,
        stop_batchnorm=10_000,
        power_iterations=500,
        subnet_depth=1,
        gradual=False,
        centpen=0
    )

    default_config['model'] = args.net
    default_config['lr'] = args.lr
    default_config['gradpen'] = args.gradpen
    default_config['centpen'] = args.centpen
    default_config['power_iterations'] = args.power_its
    default_config['gradual'] = args.gradual

    if args.dryrun:
        default_config['power_iterations'] = 1

    print('Currently evaluating -------------------------------:')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(default_config)
    print('Python version is:')
    print(sys.version)
    print(f'PyTorch version is {torch.__version__}')
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.torch.cuda.device_count()}')

    print('-----------------------------------------------------')
    # try:
    evaluation_procedure(default_config)
    #    print('Job finished. ---------------------------------------')
    # except (KeyboardInterrupt, SystemExit):
    #    raise
    # except Exception as e:
    #    print(repr(e))
    #    print('Job failed!! ----------------------------------------')
    print('-----------------------------------------------------')


def evaluation_procedure(config):
    """Train model and evaluate eigenvalues with given configuration."""
    # Setup data
    augmentations = False
    trainloader, testloader = dl.get_loaders('CIFAR10', config['batch_size'],
                                             augmentations=augmentations, normalize=True, shuffle=False)

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
    elif config['model'] == 'ResNet':
        net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    else:
        raise NotImplementedError()

    linear_classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 10))

    linear_classifier.to(**config['setup'])
    net.to(**config['setup'])
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        linear_classifier = torch.nn.DataParallel(linear_classifier)
    net.eval()

    # Optimizer and loss
    optimizer = torch.optim.SGD(linear_classifier.parameters(), lr=config['lr'],
                                momentum=0.9, weight_decay=config['weight_decay'])
    config['epochs'] = config['epochs_linear']
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 85, 95], gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Check initial model
    analyze_model(linear_classifier, trainloader, testloader, loss_fn, config)

    linear_classifier.to(**config['setup'])
    net.to(**config['setup'])
    # Train
    print('Starting training linear classifier ...')
    dl.train(linear_classifier, optimizer, scheduler, loss_fn, trainloader, config, dryrun=args.dryrun)
    # Analyze results
    print('----Results after training linear classifier ------------')
    analyze_model(linear_classifier, trainloader, testloader, loss_fn, config)
    for name, param in linear_classifier.named_parameters():
        dprint(name, param)
        param.requires_grad = False
    # Check full model
    print('----Distill learned classifier onto network ------------')
    config['epochs'] = config['epochs_distill']
    loss_distill = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 180, 240], gamma=0.2)
    dl.distill(linear_classifier, net, optimizer, scheduler, loss_distill, trainloader, config, dryrun=args.dryrun)

    # Analyze results
    analyze_model(net, trainloader, testloader, loss_fn, config)


def analyze_model(model, trainloader, testloader, loss_fn, config):
    """Get accuracy, loss, 1st order optimality and 2nd order optimality for model."""
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    model.to(**config['setup'])
    # Accuracy and loss
    print(f'Accuracy of the network on training images: {(100 * dl.get_accuracy(model, trainloader, config))} %%')
    print(f'Accuracy of the network on test images: {(100 * dl.get_accuracy(model, testloader, config))} %%')
    print(f'Loss in training is {dl.compute_loss(model, loss_fn, trainloader, config):.12f}')
    # print(f'Loss in testing is {dl.compute_loss(model, loss_fn, testloader, config):.12f}')
    # 1st order opt
    grd_train = dl.gradient_norm(trainloader, model, loss_fn, config['setup']['device'], config['weight_decay'])
    grd_test = dl.gradient_norm(testloader, model, loss_fn, config['setup']['device'], config['weight_decay'])
    print(f'Gradient norm in training is {grd_train:.12f}')
    print(f'Gradient norm in testing is {grd_test:.12f}')
    # 2nd order opt
    hessian = dl.HessianOperator(model, trainloader, loss_fn, weight_decay=config['weight_decay'], **config['setup'])
    dl.eigenvalue_analysis(hessian, method='power_method', tol=1e-8, max_iter=config['power_iterations'])
    print('EVs computed ...')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    # Check throughput
    if not args.dryrun:
        print('Checking throughput: ...')
        throughput_dict = dl.check_throughputs(model, trainloader, quiet=True, device=torch.device('cpu'))
        print(throughput_dict)


def dprint(*nargs):
    """Print only in debug mode."""
    if args.debug:
        print(*nargs)

# ### ACTUAL CODE
if __name__ == "__main__":
    run_job()
