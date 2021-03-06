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
parser.add_argument('--print', default=25, type=int)
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

    num_params = sum([p.numel() for p in net.parameters()])
    print(f'Number of params: {num_params} - number of data points: {len(trainloader.dataset)} '
          f'- ratio : {len(trainloader.dataset) / num_params * 100:.2f}%')

    ntk_matrix_before = batch_wise_ntk(net, trainloader, samplesize=args.sampling)
    ntk_matrix_before_norm = np.linalg.norm(ntk_matrix_before.flatten())
    print(f'The total norm of the NTK sample before training is {ntk_matrix_before_norm:.2f}')
    param_norm_before = np.sqrt(np.sum([p.pow(2).sum().detach().cpu().numpy() for p in net.parameters()]))
    print(f'The L2 norm of the parameter vector is {param_norm_before:.2f}')

    # Start training
    net.to(**config['setup'])

    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    loss_fn = torch.nn.CrossEntropyLoss()
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))

    """Standardized pytorch training routine."""
    ntk_norms = []
    param_norms = []
    corrs = []
    corrs_diff = []
    sims = []
    epochs = []
    logcount = [int(x) for x in np.geomspace(1, config['epochs'], 50)]

    # Do epoch -1
    ntk_matrix = batch_wise_ntk(net, trainloader, samplesize=args.sampling)
    ntk_norm = np.linalg.norm(ntk_matrix.flatten())
    print(f'The total norm of the NTK sample is {ntk_norm:.2f}')
    param_norm = np.sqrt(np.sum([p.pow(2).sum().detach().cpu().numpy() for p in net.parameters()]))
    print(f'The L2 norm of the parameter vector is {param_norm:.2f}')
    matrix_corr, corr_coeff, corr_tom = corr(ntk_matrix_before, ntk_matrix)
    print(f'The Correlation coefficient of the NTK sample from init to now is {corr_coeff:.2f}')
    print(f'The Similarity coefficient of the NTK sample from init to now is {corr_tom:.2f}')
    ntk_matrix_prev = ntk_matrix_before.copy()
    sims.append(corr_tom)
    param_norms.append(param_norm)
    ntk_norms.append(ntk_norm)
    corrs.append(corr_coeff)
    corrs_diff.append(corr_coeff)
    epochs.append(0)

    net.train()
    net.to(**config['setup'])
    for epoch in range(config['epochs']):
        # Train
        epoch_loss = 0.0
        for i, (inputs, targets) in enumerate(trainloader):
            optimizer.zero_grad()
            inputs = inputs.to(device=config['setup']['device'], dtype=config['setup']['dtype'])
            targets = targets.to(device=config['setup']['device'], dtype=torch.long)
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if args.dryrun:
                break

        if epoch in logcount:
            print(f'Epoch loss in epoch {epoch} was {epoch_loss/(i+1):.12f}')
            # Get NTK sample
            ntk_matrix = batch_wise_ntk(net, trainloader, samplesize=args.sampling)
            ntk_norm = np.linalg.norm(ntk_matrix.flatten())
            print(f'The total norm of the NTK sample is {ntk_norm:.2f}')
            param_norm = np.sqrt(np.sum([p.pow(2).sum().detach().cpu().numpy() for p in net.parameters()]))
            print(f'The L2 norm of the parameter vector is {param_norm:.2f}')
            _, corr_coeff, corr_tom = corr(ntk_matrix_before, ntk_matrix)
            print(f'The Correlation coefficient of the NTK sample from init to now is {corr_coeff:.2f}')
            print(f'The Similarity coefficient of the NTK sample from init to now is {corr_tom:.2f}')
            _, corr_coeff_diff, _  = corr(ntk_matrix_prev, ntk_matrix)
            print(f'The Correlation coefficient of the NTK sample to NTK of prev step is {corr_coeff_diff:.2f}')
            ntk_matrix_prev = ntk_matrix.copy()
            param_norms.append(param_norm)
            ntk_norms.append(ntk_norm)
            corrs.append(corr_coeff)
            corrs_diff.append(corr_coeff_diff)
            sims.append(corr_tom)
            epochs.append(epoch + 1)

            net.train()
            net.to(**config['setup'])

        scheduler.step()
        if args.dryrun:
            break
    net.eval()

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))

    param_norm_after = np.sqrt(np.sum([p.pow(2).sum().detach().cpu().numpy() for p in net.parameters()]))
    print(f'The L2 norm of the parameter vector is {param_norm_after:.2f}')

    ntk_matrix_after = batch_wise_ntk(net, trainloader, samplesize=args.sampling)
    ntk_matrix_after_norm = np.linalg.norm(ntk_matrix_after.flatten())
    print(f'The total norm of the NTK sample after training is {ntk_matrix_after_norm:.2f}')

    ntk_matrix_diff = np.abs(ntk_matrix_before - ntk_matrix_after)
    ntk_matrix_diff_norm = np.linalg.norm(ntk_matrix_diff.flatten())
    print(f'The total norm of the NTK sample diff is {ntk_matrix_diff_norm:.2f}')

    ntk_matrix_rdiff = np.abs(ntk_matrix_before - ntk_matrix_after) / (np.abs(ntk_matrix_before) + 1e-4)
    ntk_matrix_rdiff_norm = np.linalg.norm(ntk_matrix_rdiff.flatten())
    print(f'The total norm of the NTK sample relative diff is {ntk_matrix_rdiff_norm:.2f}')

    save_output(args.table_path, name='ntk', width=config['width'], num_params=num_params,
                before_norm=ntk_matrix_before_norm, after_norm=ntk_matrix_after_norm,
                diff_norm=ntk_matrix_diff_norm, rdiff_norm=ntk_matrix_rdiff_norm,
                param_norm_before=param_norm_before, param_norm_after=param_norm_after)

    # Check for file
    fname = os.path.join(args.table_path, f'table_evo_{args.net}_{args.width}.csv')

    # Read or write header
    with open(fname, 'w', newline='') as f:
        evo_writer = csv.writer(f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        evo_writer.writerow(epochs)
        evo_writer.writerow(ntk_norms)
        evo_writer.writerow(param_norms)
        evo_writer.writerow(corrs)
        evo_writer.writerow(corrs_diff)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('-----------------------------------------------------')
    print('Job finished.----------------------------------------')
    print('-----------------------------------------------------')


def corr(n1, n2):
    """Pearson Corr."""
    n1_mean = np.mean(n1)
    n2_mean = np.mean(n2)
    matrix_corr = (n1 - n1_mean) * (n2 - n2_mean) / \
        np.std(n1) / np.std(n2)
    corr_coeff = np.mean(matrix_corr)
    corr_tom = np.sum(n1 * n2) / np.sqrt(np.sum(n1 * n1) * np.sum(n2 * n2))
    return matrix_corr, corr_coeff, corr_tom


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
    fname = os.path.join(out_dir, f'table_evo_{args.net}_{name}.csv')
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
        # Compute gradients per image sample per logit
        image_grads = []
        for n in range(samplesize):
            output = net(inputs[n:n + 1, :, :, :]).squeeze()
            logit_dim = output.shape[0]
            D_ft = []
            for l in range(logit_dim):
                net.zero_grad()
                D_ft.append(torch.autograd.grad(output[l], net.parameters(), only_inputs=True, retain_graph=True))
            image_grads.append(D_ft)
        length_last = sum([p.numel() for p in image_grads[-1][-1]])
        print(f'Gradients computed in dim (samples x logits x params) :'
              f' ({samplesize} x {logit_dim} x {length_last})')
        ntk_sample = []

        for ni in range(samplesize):                # image 1
            ntk_row = []
            for nj in range(samplesize):            # image 2
                ntk_entry = np.empty((logit_dim, logit_dim))
                for i in range(logit_dim):          # iterate over logits
                    for j in range(logit_dim):
                        prod = 0
                        for p1, p2 in zip(image_grads[ni][i], image_grads[nj][j]):
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
