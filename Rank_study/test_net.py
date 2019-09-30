################################################################################
#
# test_net.py
# Main method for testing network performance
# September 2019
#
################################################################################

import argparse
import datetime
from model_classes import ResNet18
from utils import AttackPGD, test
import torch
import torchvision
import torchvision.transforms as transforms


def main():
    print("\n_________________________________________________\n")
    print("Starting main method in test_net.py")
    print("Time now: ", datetime.datetime.now())

    parser = argparse.ArgumentParser(description='PyTorch rank study')
    parser.add_argument('--model', default='ResNet18', type=str, help='model for training')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset')
    parser.add_argument('--output', default='output_default', type=str, help='output subdirectory')
    parser.add_argument('--eval_attack_steps', default=20, type=int,
                        help='number of attack steps for PGD during testing')
    parser.add_argument('--checkpoint_folder', default='checkpoint_default', type=str, help='checkpoint folder name')
    parser.add_argument('--model_path', default='', type=str, help='where is the pretrained model?')
    parser.add_argument('--clip_to', default=1.0, type=float, help='clipping for singular values in rank regularizer')
    parser.add_argument('--adversarial', action='store_true', help='adversarial training?')
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                          transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                             transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128)



    config = {
        'epsilon': 1.0 / 255,
        'num_steps': args.eval_attack_steps,
        'step_size': .25 / 255,
    }

    # Define model
    if args.model == 'ResNet18':
        basic_net = ResNet18(num_channels=3)
        dict = torch.load(args.model_path, map_location='cpu')
        basic_net.load_state_dict(dict['net'])
    else:
        print('Model not implemented.')
        return

    net = AttackPGD(basic_net, config, args.adversarial)
    net = net.to(device)

    params = {'model path': args.model_path, 'dataset': 'train'}

    print('\n\t==> Testing...')
    train_natural_acc, train_robust_acc = test(net, basic_net, trainloader, config,
                                             params, args.output, args.adversarial,
                                             evaluate=True, log=True)

    params = {'model path': args.model_path, 'dataset': 'test'}

    test_natural_acc, test_robust_acc = test(net, basic_net, testloader, config,
                                             params, args.output, args.adversarial,
                                             evaluate=True, log=True)
    print('\tTrain Acc: ', train_natural_acc, ' Val Natural: ', test_natural_acc, ' Val Robust: ', test_robust_acc, '\n')


    print("ending  main method in test_net.py")
    print("Time now: ", datetime.datetime.now())
    print("\n_________________________________________________\n")

    return


if __name__ == "__main__":
    main()