############################################################
#
# utils.py
# utility functions
# September 2019
#
############################################################

import matplotlib as mpl
# if os.environ.get('DISPLAY', '') == '':
print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os


my_punc = '!"#$%&\'()*+/;<=>?@[\\]^`{|}~'


def adjust_learning_rate(optimizer, epoch, lr, lr_schedule, lr_factor):
    if epoch in lr_schedule:
        print('(lr drop)')
        lr *= lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def test_log(out_dir, params, results):
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, 'table.csv')

    table = str.maketrans({key: None for key in my_punc})
    paramstr = str(params).translate(table)
    res_str = str(results).translate(table)

    with open(fname, 'a') as f:
        f.write('\n' + paramstr)
        f.write('\n' + res_str)
        f.write('\n')

    print('\tTest results logged in ' + out_dir + '.')


def data_log(out_dir, params):
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, 'table.csv')

    table = str.maketrans({key: None for key in my_punc})
    paramstr = str(params).translate(table)

    with open(fname, 'a') as f:
        f.write('\n------------------------------------------\n')
        f.write('\tdata re-processed with parameters:\n')
        f.write('\t' + paramstr + '\n')

    print('Data processing logged.')


def plot_loss(data, epochs, model, outdir):
    outstr = outdir+'/e'+str(epochs)+str(model)+'loss'
    outstr = outstr.replace(".", "")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(data, 'b', label='Test')
    # ax.set_ylim([-0.001, 0.05])
    ax.set_title('Loss')
    ax.legend()
    fig.savefig(outstr)


def plot_acc(data, epochs, model, outdir):
    outstr = outdir+'/e'+str(epochs)+str(model)+'acc'
    outstr = outstr.replace(".", "")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(data[0, :], 'k', label='Train')
    ax.plot(data[1, :], 'b', label='Natural Validation')
    ax.plot(data[2, :], 'r', label='Robust Validation')
    ax.set_title('Accuracy')
    ax.legend()
    fig.savefig(outstr)


class AugmentedDataset(data.Dataset):

    def __init__(self, trainset1, trainset2=None):
        super(AugmentedDataset, self).__init__()
        self.trainset1 = trainset1
        self.trainset2 = trainset2

    def __getitem__(self, index):
        if self.trainset2:
            if index > self.trainset1.__len__() - 1:
                return self.trainset2.__getitem__(index - self.trainset1.__len__())[0], \
                       self.trainset2.__getitem__(index - self.trainset1.__len__())[1], \
                       True
            else:
                return self.trainset1.__getitem__(index)[0], \
                       self.trainset1.__getitem__(index)[1], \
                       False
        else:
            return self.trainset1.__getitem__(index)[0], \
                   self.trainset1.__getitem__(index)[1], \
                   False


    def __len__(self):
        if self.trainset2:
            return self.trainset1.__len__() + self.trainset2.__len__()
        else:
            return self.trainset1.__len__()


class AttackPGD(nn.Module):
    def __init__(self, basic_net, config, attack):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.attack = attack

    def forward(self, inputs, targets):
        if not self.attack:
            return self.basic_net(inputs), inputs
        x = inputs.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(self.basic_net(x), targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0.0, 1.0)
        return self.basic_net(x), x


def max_operator_norm(filter, inp_shape, clip_to):
    # compute the singular values using FFT
    # first compute the transforms for each pair of input and output channels
    transform_coeff = np.fft.fft2(filter, inp_shape, axes=[0, 1])

    # now, for each transform coefficient, compute the singular values of the
    # matrix obtained by selecting that coefficient for
    # input-channel/output-channel pairs
    U, D, V = np.linalg.svd(transform_coeff, compute_uv=True, full_matrices=False)
    D_clipped = np.minimum(D, clip_to)
    # D_clipped = np.where(D<clip_to, 0, D)

    if filter.shape[2] > filter.shape[3]:
        clipped_transform_coeff = np.matmul(U, D_clipped[..., None] * V)
    else:
        clipped_transform_coeff = np.matmul(U * D_clipped[..., None, :], V)
    clipped_filter = np.fft.ifft2(clipped_transform_coeff, axes=[0, 1]).real
    args = [range(d) for d in filter.shape]
    # print(clipped_filter.shape, args)
    return clipped_filter[np.ix_(*args)]


def max_rank(net, clip_to):
    print('Clipping network...')

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    resnet_sizes = {'conv1.weight': [32, 32],
                    'layer1.0.conv1.weight': [56, 56],
                    'layer1.0.conv2.weight': [56, 56],
                    'layer1.1.conv1.weight': [56, 56],
                    'layer1.1.conv2.weight': [56, 56],
                    'layer2.0.conv1.weight': [56, 56],
                    'layer2.0.conv2.weight': [28, 28],
                    'layer2.1.conv1.weight': [28, 28],
                    'layer2.1.conv2.weight': [28, 28],
                    'layer3.0.conv1.weight': [28, 28],
                    'layer3.0.conv2.weight': [14, 14],
                    'layer3.1.conv1.weight': [14, 14],
                    'layer3.1.conv2.weight': [14, 14],
                    'layer4.0.conv1.weight': [14, 14],
                    'layer4.0.conv2.weight': [7, 7],
                    'layer4.1.conv1.weight': [7, 7],
                    'layer4.1.conv2.weight': [7, 7]}

    for n, p in net.named_parameters():
        if n == 'conv1.weight':
            continue
        elif 'conv' in n and 'weight' in n:
            filter = p.permute(2, 3, 0, 1).detach().cpu().numpy()
            new_p = torch.FloatTensor(max_operator_norm(filter, resnet_sizes[n], clip_to))
            p.data = new_p.permute(2, 3, 0, 1).to(device)


def clip_operator_norm(filter, inp_shape, clip_to):
    # compute the singular values using FFT
    # first compute the transforms for each pair of input and output channels
    transform_coeff = np.fft.fft2(filter, inp_shape, axes=[0, 1])

    # now, for each transform coefficient, compute the singular values of the
    # matrix obtained by selecting that coefficient for
    # input-channel/output-channel pairs
    U, D, V = np.linalg.svd(transform_coeff, compute_uv=True, full_matrices=False)
    # D_clipped = np.minimum(D, clip_to)
    D_clipped = np.where(D<clip_to, 0, D)

    if filter.shape[2] > filter.shape[3]:
        clipped_transform_coeff = np.matmul(U, D_clipped[..., None] * V)
    else:
        clipped_transform_coeff = np.matmul(U * D_clipped[..., None, :], V)
    clipped_filter = np.fft.ifft2(clipped_transform_coeff, axes=[0, 1]).real
    args = [range(d) for d in filter.shape]
    # print(clipped_filter.shape, args)
    return clipped_filter[np.ix_(*args)]


def clip_network(net, clip_to):
    print('Clipping network...')

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    resnet_sizes = {'conv1.weight': [32, 32],
                    'layer1.0.conv1.weight': [56, 56],
                    'layer1.0.conv2.weight': [56, 56],
                    'layer1.1.conv1.weight': [56, 56],
                    'layer1.1.conv2.weight': [56, 56],
                    'layer2.0.conv1.weight': [56, 56],
                    'layer2.0.conv2.weight': [28, 28],
                    'layer2.1.conv1.weight': [28, 28],
                    'layer2.1.conv2.weight': [28, 28],
                    'layer3.0.conv1.weight': [28, 28],
                    'layer3.0.conv2.weight': [14, 14],
                    'layer3.1.conv1.weight': [14, 14],
                    'layer3.1.conv2.weight': [14, 14],
                    'layer4.0.conv1.weight': [14, 14],
                    'layer4.0.conv2.weight': [7, 7],
                    'layer4.1.conv1.weight': [7, 7],
                    'layer4.1.conv2.weight': [7, 7]}

    for n, p in net.named_parameters():
        if n == 'conv1.weight':
            continue
        elif 'conv' in n and 'weight' in n:
            filter = p.permute(2, 3, 0, 1).detach().cpu().numpy()
            new_p = torch.FloatTensor(clip_operator_norm(filter, resnet_sizes[n], clip_to))
            p.data = new_p.permute(2, 3, 0, 1).to(device)


def get_nuclear_norm(p):
    complex_p = torch.stack((p, torch.zeros_like(p)))
    complex_p = complex_p.permute(1, 2, 3, 4, 0)
    f = torch.fft(complex_p, 3)
    s = torch.svd(f, compute_uv=False)
    return torch.sum(s)


def train(net, trainloader, optimizer):

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    # Set net to train and zeros stats
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    iterator = tqdm(trainloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets, flip) in enumerate(iterator):

        # ### Debug
        # if batch_idx > 0:
        #     continue
        # ###

        inputs, targets, flip = inputs.to(device), targets.to(device), flip.to(device)
        optimizer.zero_grad()
        outputs, pert_inputs = net(inputs, targets)
        p = F.softmax(outputs, dim=1)
        flip = flip.view(-1, 1).float()
        softmax_output = torch.mul(p, 1-flip) + torch.mul(1-p, flip)
        softmax_output = torch.clamp(softmax_output, 1e-32, 1)
        log_softmax_output = torch.log(softmax_output)
        loss = criterion(log_softmax_output, targets)

        loss.backward()

        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += (1-flip).sum().item()
        correct += ((1-flip).byte().view(1, -1) * predicted.eq(targets)).sum().item()

    acc = 100. * correct / total

    return train_loss, acc


def test(net, basic_net, testloader, config, params, out_dir, attack, evaluate=False, log=False):

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net.eval()
    adv_correct = 0
    natural_correct = 0
    total = 0
    results = {}

    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):

            # ### Debug
            # if batch_idx > 0:
            #     continue
            # ###

            inputs, targets = inputs.to(device), targets.to(device)
            natural_outputs = basic_net(inputs)
            if attack:
                if evaluate:
                    eval_net = AttackPGD(basic_net, config, attack=True)
                    adv_outputs, pert_inputs = eval_net(inputs, targets)
                else:
                    adv_outputs, pert_inputs = net(inputs, targets)
            else:
                adv_outputs = natural_outputs
            _, adv_predicted = adv_outputs.max(1)
            _, natural_predicted = natural_outputs.max(1)
            natural_correct += natural_predicted.eq(targets).sum().item()
            adv_correct += adv_predicted.eq(targets).sum().item()
            total += targets.size(0)

    robust_acc = 100. * adv_correct / total
    natural_acc = 100. * natural_correct / total
    results['Robust acc'] = robust_acc
    results['Clean acc'] = natural_acc

    if log:
        test_log(out_dir, params, results)

    return natural_acc, robust_acc


def get_data_sets(dataset, poormin=False):

    if dataset == "MNIST":

        # Clean data
        channels = 1
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset_clean = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                              transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                             transform=transform_test)

        # Flipped
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor()
        ])
        trainset_flip = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                              transform=transform_train)
    elif dataset == "CIFAR10":

        # Clean data
        channels = 3
        if not poormin:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor()
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset_clean = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                              transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                             transform=transform_test)

        # Flipped
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor()
        ])
        trainset_flip = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                              transform=transform_train)
    else:
        print("Dataset not yet implemented. Terminating.")
        return

    # Combined
    trainset = AugmentedDataset(trainset_clean, trainset_flip)

    if not poormin:
        trainloader = torch.utils.data.DataLoader(AugmentedDataset(trainset_clean), batch_size=128)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128)

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128)

    return trainloader, testloader, channels

