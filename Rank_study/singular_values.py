############################################################
#
# singular_values.py
# compute the singular values of convolutional layers
# August 2019
#
############################################################

import numpy as np
import os
import torch
import torchvision
import argparse
from model_classes import ResNet18
import datetime


def get_singular_values_conv(filter, input_size):
    transform_coeff = np.fft.fft2(filter, input_size, axes=[0, 1])
    s =  np.linalg.svd(transform_coeff, compute_uv=False)
    return np.flip(np.sort(s.flatten()), 0)


def save_out(dest, obj):
    out_dir = os.path.join(*(dest.split('/')[:-1]))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    np.save(dest, obj)


def main():
    print("\n_________________________________________________\n")
    print("Starting main method in singular_values.py")
    print("Time now: ", datetime.datetime.now())

    parser = argparse.ArgumentParser(description='Rank study: plot generation')
    parser.add_argument('--model', default='ResNet18', type=str, help='Which model')
    parser.add_argument('--model_path', default='', type=str, help='where is the pretrained model?')
    parser.add_argument('--save_name', default='untrained', type=str, help='Name to save as')
    args = parser.parse_args()
    model = args.model

    if model == 'ResNet18':

        resnet = ResNet18(num_channels=3)
        if args.model_path is not '':
            dict = torch.load(args.model_path, map_location='cpu')
            resnet.load_state_dict(dict['net'])

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

    else:
        print('Singular value computation not yet impolemented for model: ', model)
        return

    net = resnet
    for n, p in net.named_parameters():
        if 'conv' in n and 'weight' in n:
            out_str = os.path.join(model, args.save_name, n)

            if os.path.isfile(out_str+'.npy'):
                print(out_str)
                continue
            p = p.permute(2, 3, 0, 1)
            singular_values = get_singular_values_conv(p.detach().numpy(), resnet_sizes[n])
            save_out(out_str, singular_values)
            print(len(singular_values))

    print("ending  main method in singular_values.py")
    print("Time now: ", datetime.datetime.now())
    print("\n_________________________________________________\n")
    return

if __name__ == '__main__':
    main()