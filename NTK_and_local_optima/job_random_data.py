"""Check random data in wide networks."""

"""
"""
import argparse


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import dl_myths as dl
from pytorch_cifar.models import ResNet18, ResNet, WideResNet, BasicBlock
from collections import defaultdict, OrderedDict
import datetime

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
config[''] = 10_000
config['stop_batchnorm'] = 750
config['full_batch'] = True
config['setup'] = dict(device=device,dtype=dtype)

"""

parser = argparse.ArgumentParser(description='Run a single job_random_data experiment')
parser.add_argument('--id', default=None, type=int, help='job id from slurm')

parser.add_argument('--width', default=1, type=int, help='ResNet Width')
parser.add_argument('--aug', default=1, type=int, help='amount of augmentation')
parser.add_argument('--do_aug', default=1, type=int, help='actual augmentation or extension of data')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--nosave', action='store_false')
args = parser.parse_args()


def run_job():
    """Just run a single job from a list of ids."""
    #
    print(f'EVALUATING RANDOMDATA EXPERIMENT ID {args.id} ---------------:')
    default_config = dict(
        batch_size=128,
        model='ResNet',
        setup=dict(device=torch.device('cuda:0'), dtype=torch.float),
        epochs=500,
        print_loss=10,
        lr=0.01,
        weight_decay=5e-4,
        full_batch=False,
        switch_to_gd=10_000,
        stop_batchnorm=10_000,
        save_models=args.nosave,
    )

    # Run job by id if ids are given like for SLURM
    if args.id is not None:
        id_counter = 0
        for model in ['ResNetWide1', 'ResNetWide2', 'ResNetWide5', 'ResNetWide7']:
            if model == 'ResNetWide10':
                default_config['batch_size'] = 32
                default_config['lr'] = 0.05
            for data_augmentation in [None, 'default', 'alot']:
                for extra_data in ['do_augmentation', 'add_extra_data']:
                    id_counter += 1
                    if id_counter == args.id:
                        default_config['model'] = model
                        default_config['data_augmentation'] = data_augmentation
                        default_config['extra_data'] = extra_data
                        task(default_config)
    else:
        default_config['model'] = 'ResNetWide' + str(args.width)
        default_config['data_augmentation'] = None if args.aug == 0 else ('default' if args.aug == 1 else 'alot')
        default_config['extra_data'] = 'do_augmentation' if args.do_aug == 1 else 'add_extra_data'
        task(default_config)


def task(default_config):
    """Task wrapper."""
    print('Currently evaluating -------------------------------:')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(default_config)
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.torch.cuda.device_count()}')
    try:
        evaluation_procedure(default_config)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(repr(e))
    print('-----------------------------------------------------')
    print('Job finished.----------------------------------------')
    print('-----------------------------------------------------')


def evaluation_procedure(config):
    """Train model and evaluate eigenvalues with given configuration."""
    # Setup data
    trainset = torchvision.datasets.FakeData(size=50_000, image_size=(3, 32, 32), num_classes=10,
                                             transform=transforms.ToTensor(), random_offset=0)
    cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    print(f'Data mean is {data_mean}, data std is  {data_std}')
    if config['data_augmentation'] is None:
        padding = 0
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)])
    elif config['data_augmentation'] == 'default':
        padding = 2
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=padding, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)])
    elif config['data_augmentation'] == 'alot':
        padding = 4
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=padding, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)])

    trainset = torchvision.datasets.FakeData(size=50_000, image_size=(3, 32, 32), num_classes=10,
                                             transform=transform, random_offset=0)
    testset = torchvision.datasets.FakeData(size=1_000, image_size=(3, 32, 32), num_classes=10,
                                            transform=transforms.ToTensor(), random_offset=267914296)  # fib42
    aug = max((padding * 2) ** 2 * 2, 1)
    aug_datapoints = len(trainset) * aug
    if config['extra_data'] == 'add_extra_data':
        padding = 0
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)])
        trainset = torchvision.datasets.FakeData(size=aug_datapoints, image_size=(3, 32, 32), num_classes=10,
                                                 transform=transform, random_offset=0)
        config['epochs'] = max(config['epochs'] // aug, 20)
        print(f'Effective epochs reduced to {config["epochs"]}')
    elif config['extra_data'] == 'do_augmentation':
        pass

    num_workers = torch.get_num_threads() if torch.get_num_threads() > 0 else 0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                              shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'],
                                             shuffle=True, num_workers=num_workers)

    # Setup Network
    if config['model'] == 'ResNetWide1':
        net = ResNet18()
    elif config['model'] == 'ResNetWide2':
        net = WideResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, widen_factor=2)
    elif config['model'] == 'ResNetWide5':
        net = WideResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, widen_factor=5)
    elif config['model'] == 'ResNetWide7':
        net = WideResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, widen_factor=7)
    else:
        net = WideResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, widen_factor=10)

    net.to(**config['setup'])
    net = torch.nn.DataParallel(net)
    net.eval()

    num_params = sum([p.numel() for p in net.parameters()])
    print(f'Number of params: {num_params} - number of data points: {len(trainloader.dataset)} '
          f'- ratio : {len(trainloader.dataset) / num_params * 100:.2f}%')
    aug_datapoints = len(trainloader.dataset) * max((padding * 2) ** 2 * 2, 1)
    print(f'Number of params: {num_params} - number of aug. data points: {aug_datapoints}'
          f'- ratio : {aug_datapoints/num_params*100:.2f}%')

    # Optimizer and loss
    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])

    if config['extra_data'] == 'add_extra_data':
        scheduling = [max(100 // aug, 5), max(250 // aug, 10), max(350 // aug, 15)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduling,
                                                         gamma=0.1)
        print(f'New scheduling is {scheduling}.')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250, 350], gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    full_batch = config['full_batch']

    print('Start training ...')
    dl.train(net, optimizer, scheduler, loss_fn, trainloader, config, path=None, dryrun=args.dryrun)
    if config['save_models']:
        torch.save(net.state_dict(), 'models/' + config['model'] +
                   '_random_labels_' + str(config['weight_decay']) + str(config['data_augmentation']) + '.pth')

    print(f'Accuracy of the network on training images: {(100 * dl.get_accuracy(net, trainloader, config))} %%')
    print(f'Accuracy of the network on test images: {(100 * dl.get_accuracy(net, testloader, config))} %%')
    print(f'Loss in training is {dl.compute_loss(net, loss_fn, trainloader, config):.12f}')
    print(f'Loss in testing is {dl.compute_loss(net, loss_fn, testloader, config):.12f}')
    grd_train = dl.gradient_norm(trainloader, net, loss_fn, config['setup']['device'], config['weight_decay'])
    grd_test = dl.gradient_norm(testloader, net, loss_fn, config['setup']['device'], config['weight_decay'])
    print(f'Gradient norm in training is {grd_train:.12f}')
    print(f'Gradient norm in testing is {grd_test:.12f}')

# ### ACTUAL CALL

run_job()
