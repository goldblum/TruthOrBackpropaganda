from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from tqdm import tqdm
from models import ResNet18, fixup_resnet20, MobileNetV2, densenet_cifar
from output_module import save_output
import random

from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lr_schedule', type=int, nargs='+',
                    default=[25, 35, 40, 45], help='Decrease learning rate at these epochs.')
parser.add_argument('--lr_factor', default=0.1, type=float, help='factor by which to decrease lr')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs for training')
parser.add_argument('--resume_path', default='', type=str, help='output directory')
parser.add_argument('--resume_epoch', action='store_true', help='continue with the same epoch number')
parser.add_argument('--resume_optimizer', action='store_true', help='continue with the same optimizer')
parser.add_argument('--output', default='', type=str, help='output directory')
parser.add_argument('--model', default='ResNet18', type=str, help='model for training')
parser.add_argument('--val_period', default=10, type=int, help='print every __ epoch')
parser.add_argument('--save_period', default=300, type=int, help='save every __ epoch')
parser.add_argument('--eval', action='store_true', help='dont train')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='name of dataset to train/test on')
parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay')
parser.add_argument('--norm_bias', default=0.0, type=float, help='attractor for squared l2 norm of weights')
parser.add_argument('--debug', action='store_true', help='run for one epoch')
parser.add_argument('--subset_size', default=50000, type=int, help='amount of training data to use')
parser.add_argument('--no_aug', action='store_true', help='Dont use data augmentation')
parser.add_argument('--no_norm', action='store_true', help='Dont use data normalization')
parser.add_argument('--width', default=2048, type=int, help='MLP width')
parser.add_argument('--runs', default=1, type=int, help='No. of runs')
args = parser.parse_args()
if args.debug:
    args.epochs = 1
    args.val_period = 1
    args.save_period = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def parameters_Norm(optimizer):
    # Computes the squared L2 norm of parameters
    return np.sum([np.sum(np.array([torch.sum(p*p).detach().cpu().numpy() for p in group['params']]))
                   for group in optimizer.param_groups])


def BN_higherNorm_2norm(optimizer, norm_bias, weight_decay):
    factor = 2.0*parameters_Norm(optimizer)-2.0*norm_bias
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if weight_decay != 0:
                d_p.add_(weight_decay*factor*p.data)


def BN_higherNorm_1norm(optimizer, norm_bias, weight_decay):
    diff = parameters_Norm(optimizer)-norm_bias
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if diff > 0:
                d_p.add_(weight_decay*p.data)
            elif diff < 0:
                d_p.add_(-weight_decay*p.data)


def flatten_parameters_relative(initialization, trained_net, normalization=None):
    initialization_layers = []
    trained_net_layers = []
    layer_norm = 1.0
    for init, param in zip(initialization.parameters(), trained_net.parameters()):
        current_initialization_layer = init.detach().cpu().numpy().flatten()
        current_trained_net_layer = param.detach().cpu().numpy().flatten()
        if normalization and len(param.size()) > 1:
            layer_norm = np.linalg.norm(current_initialization_layer, normalization)
        current_initialization_layer = current_initialization_layer/layer_norm
        current_trained_net_layer = current_trained_net_layer/layer_norm
        initialization_layers.append(current_initialization_layer)
        trained_net_layers.append(current_trained_net_layer)
    return np.concatenate(initialization_layers), np.concatenate(trained_net_layers)


def change_in_pNorm(initialization, trained_net, p='inf', normalization=None):
    initialization_params, trained_net_params = flatten_parameters_relative(
        initialization, trained_net, normalization=normalization)
    diff = trained_net_params - initialization_params
    return np.linalg.norm(diff, p)


def pNorm(initialization, trained_net, p='inf', normalization=None):
    initialization_params, trained_net_params = flatten_parameters_relative(
        initialization, trained_net, normalization=normalization)
    return np.linalg.norm(trained_net_params, p)


def adjust_learning_rate(optimizer, epoch, lr):
    if epoch in args.lr_schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_factor


print('==> Preparing data..')

if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
    if args.model == 'ResNet20_FIXUP' or args.no_norm==False:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])

if args.no_aug or args.dataset == 'MNIST':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

sampler = torch.utils.data.SubsetRandomSampler(random.sample(range(50000), args.subset_size))

if args.dataset == 'CIFAR10':
    trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=0, sampler=sampler)
    testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    num_classes = 10
elif args.dataset == 'CIFAR100':
    trainset = torchvision.datasets.CIFAR100(root='~/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False, num_workers=2, sampler=sampler)
    testset = torchvision.datasets.CIFAR100(root='~/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    num_classes = 100
elif args.dataset == 'MNIST':
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    sampler = torch.utils.data.SubsetRandomSampler(random.sample(range(60000), args.subset_size))
    trainset = torchvision.datasets.MNIST(root='~/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2, sampler=sampler)
    testset = torchvision.datasets.MNIST(root='~/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    num_classes = 10

criterion = nn.CrossEntropyLoss()


def init_model():
    print('==> Building model..'+args.model)
    if args.model == 'ResNet18':
        net = ResNet18(num_classes=num_classes)
    elif args.model == 'MLP':
        # 4-layer MLP
        input_dim = 3072 if ('CIFAR' in args.dataset) else 784
        width = args.width
        net = torch.nn.Sequential(OrderedDict([
                                 ('flatten', torch.nn.Flatten()),
                                 ('linear0', torch.nn.Linear(input_dim, width)),
                                 ('relu0', torch.nn.ReLU()),
                                 ('linear1', torch.nn.Linear(width, width)),
                                 ('relu1', torch.nn.ReLU()),
                                 ('linear2', torch.nn.Linear(width, width)),
                                 ('relu2', torch.nn.ReLU()),
                                 ('linear3', torch.nn.Linear(width, num_classes))]))
    elif args.model == 'DenseNet':
        net = densenet_cifar(num_classes=num_classes)
    elif args.model == 'MobileNetV2':
        net = MobileNetV2(num_classes=num_classes)
    elif args.model == 'ResNet20_FIXUP':
        net = fixup_resnet20(num_classes=num_classes)
    else:
        raise ValueError('shitty args.model name')
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    return net


def train(epoch, optimizer, net):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iterator = tqdm(trainloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        BN_higherNorm_1norm(optimizer, args.norm_bias, args.weight_decay)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))
        if args.debug:
            break
    if (epoch+1) % args.save_period == 0:
        print('\nSaving..')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        path_name = './checkpoint/'+args.dataset+'_'+args.output+'/'
        if not os.path.isdir(path_name):
            os.makedirs(path_name, )
        if not args.debug:
            torch.save(state, path_name+'normBias='+str(args.norm_bias) +
                       '_weightDecay='+str(args.weight_decay)+'_epoch='+str(epoch)+'.t7')

    acc = 100.*correct/total
    print('\nEpoch: %d' % epoch)
    print('Train acc:', acc)
    return train_loss / (batch_idx + 1)


def test(epoch, optimizer, net):
    net.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))
            if args.debug:
                break
    acc = 100.*correct/total
    print('Val acc:', acc)
    return acc, test_loss / (batch_idx + 1)


def main():
    test_accs = []
    train_losses = []
    test_losses = []
    weight_norms = []

    for run in range(args.runs):
        lr = args.lr
        net = init_model()
        optimizer = optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=0.0)
        if not args.eval:
            for epoch in range(start_epoch, args.epochs):
                adjust_learning_rate(optimizer, epoch, lr)
                train_loss = train(epoch, optimizer, net)
                train_losses.append(train_loss)

            test_acc, test_loss = test(epoch, optimizer, net)
            test_accs.append(test_acc)
            test_losses.append(test_loss)
        weight_norm = pNorm(net, net, p=2, normalization=None)
        weight_norms.append(weight_norm)
    if not args.debug:
        save_output('./checkpoint/'+args.dataset+'_'+args.output+'/', args.dataset, args.model,
                    args.norm_bias, args.weight_decay, np.mean(train_losses),  np.mean(test_losses),
                    np.mean(test_accs), np.mean(weight_norms))
    else:
        print('./checkpoint/'+args.dataset+'_'+args.output+'/', args.dataset, args.model,
              args.norm_bias, args.weight_decay, np.mean(train_losses),  np.mean(test_losses),
              np.mean(test_accs), np.mean(weight_norms))


if __name__ == '__main__':
    main()
