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
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--subnet_depth', default=1, type=int)
parser.add_argument('--alg', default='SGD', type=str)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--width', default=3072, type=int)

# debug
parser.add_argument('--debug', action='store_true')
parser.add_argument('--dryrun', action='store_true')
args = parser.parse_args()


def run_job():
    """Wrap around experiment configs."""
    default_config = dict(
        batch_size=128,
        model='MLP',
        setup=dict(device=torch.device('cuda:0'), dtype=torch.float),
        epochs=800,
        print_loss=10,
        lr=0.01,
        weight_decay=0,
        full_batch=False,
        switch_to_gd=400,
        stop_batchnorm=90,
        power_iterations=500,
        subnet_depth=1
    )

    default_config['model'] = args.net
    default_config['subnet_depth'] = args.subnet_depth
    default_config['lr'] = args.lr
    if args.alg == 'GD':
        default_config['full_batch'] = 1
        default_config['batch_size'] = 1024
        default_config['lr'] *= default_config['batch_size'] / 50_000  # only cifar

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
                                             augmentations=augmentations, normalize=args.normalize, shuffle=False)

    class Restrict(torch.nn.Module):
        def __init__(self, subrank):
            super(Restrict, self).__init__()
            self.shape = int(subrank)

        def forward(self, x):
            return x[:, :self.shape]

    if config['model'] == 'MLP':

        fullnet = torch.nn.Sequential(OrderedDict([('flatten', torch.nn.Flatten()),
                                                   ('linear0', torch.nn.Linear(3072, args.width)),
                                                   ('relu0', torch.nn.ReLU()),
                                                   ('linear1', torch.nn.Linear(args.width, args.width)),
                                                   ('relu1', torch.nn.ReLU()),
                                                   ('linear2', torch.nn.Linear(args.width, args.width)),
                                                   ('relu2', torch.nn.ReLU()),
                                                   ('linear3', torch.nn.Linear(args.width, 10))]))
        # breakpoint()
        subnet = torch.nn.Sequential(torch.nn.Flatten(), Restrict(args.width),
                                     *list(fullnet.children())[-config['subnet_depth']:])
    else:
        raise NotImplementedError()

    subnet.to(**config['setup'])
    fullnet.to(**config['setup'])
    if torch.cuda.device_count() > 1:
        subnet = torch.nn.DataParallel(subnet)
        fullnet = torch.nn.DataParallel(fullnet)
    subnet.eval()
    fullnet.eval()

    # Optimizer and loss
    optimizer = torch.optim.SGD(subnet.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 200, 400, 600, 700], gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Check initial model
    analyze_model(subnet, trainloader, testloader, loss_fn, config)

    subnet.to(**config['setup'])
    fullnet.to(**config['setup'])
    # Train
    print('Starting training subnet ...........................................')
    dl.train(subnet, optimizer, scheduler, loss_fn, trainloader, config, dryrun=args.dryrun)

    # Analyze results
    print('----Results after training subnet -----------------------------------------------------------')
    analyze_model(subnet, trainloader, testloader, loss_fn, config)
    for name, param in subnet.named_parameters():
        dprint(name, param)
    # Check full model
    print('----Extend to full model and check local optimality -----------------------------------------')
    # assert all([p1 is  p2 for (p1, p2) in zip(fullnet[-1].parameters(), subnet.parameters())])
    bias_first = True
    bias_offset = 2
    for name, param in fullnet.named_parameters():
        if all([param is not p for p in subnet.parameters()]):
            dprint(f'Currently setting {name}')
            if 'weight' in name:
                torch.nn.init.eye_(param)
                dprint(f'{name} set to Id.')
            elif 'bias' in name:
                if bias_first:
                    torch.nn.init.constant_(param, bias_offset)
                    bias_first = False
                    dprint(f'{name} set to 1.')
                else:
                    torch.nn.init.constant_(param, 0)
                    dprint(f'{name} set to 0.')
                    # if normalize=False, input will be in [0,1] so no bias is necessary
            elif 'conv.weight' in name:
                torch.nn.init.dirac_(param)
                dprint(f'{name} set to dirac.')
        else:
            if 'linear3.bias' in name:
                Axb = subnet(bias_offset * torch.ones(1, 3072, **config['setup'])).detach().squeeze()
                param.data -= Axb - param.data
                dprint(f'{name} set to b - Ax')
    print('Model extended to full model.')
    for name, param in fullnet.named_parameters():
        dprint(name, param)
    # Analyze results
    analyze_model(fullnet, trainloader, testloader, loss_fn, config)
    # Finetune
    print('Finetune full net .................................................')
    config['full_batch'] = False
    optimizer = torch.optim.SGD(subnet.parameters(), lr=1e-4, momentum=0.9, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 200, 400, 600, 700], gamma=0.1)
    dl.train(fullnet, optimizer, scheduler, loss_fn, trainloader, config, dryrun=args.dryrun)
    analyze_model(fullnet, trainloader, testloader, loss_fn, config)


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
    # grd_test = dl.gradient_norm(testloader, model, loss_fn, config['setup']['device'], config['weight_decay'])
    print(f'Gradient norm in training is {grd_train:.12f}')
    # print(f'Gradient norm in testing is {grd_test:.12f}')
    # 2nd order opt
    hessian = dl.HessianOperator(model, trainloader, loss_fn, weight_decay=config['weight_decay'], **config['setup'])
    dl.eigenvalue_analysis(hessian, method='power_method', tol=0.0, max_iter=config['power_iterations'], quiet=True)
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
