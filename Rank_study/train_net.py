################################################################################
#
# train_net.py
# Main method for training
# September 2019
#
################################################################################

import argparse
import datetime
from model_classes import ResNet18
import torch
import torch.optim as optim
import sys
import os
from utils import adjust_learning_rate, AttackPGD, clip_network, train, test, get_data_sets, max_rank

def main():
    print("\n_________________________________________________\n")
    print("Starting main method in train_net.py")
    print("Time now: ", datetime.datetime.now())

    parser = argparse.ArgumentParser(description='PyTorch rank study')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', nargs='+', default=[15, 30, 40], type=int, help='how often to decrease lr')
    parser.add_argument('--lr_factor', default=0.1, type=float, help='factor by which to decrease lr')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs for training')
    parser.add_argument('--model', default='ResNet18', type=str, help='model for training')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset')
    parser.add_argument('--val_period', default=5, type=int, help='print every __ epoch')
    parser.add_argument('--output', default='output_default', type=str, help='output subdirectory')
    parser.add_argument('--attack_steps', default=7, type=int, help='number of attack steps for PGD during testing')
    parser.add_argument('--eval_attack_steps', default=10, type=int,
                        help='number of attack steps for PGD during testing')
    parser.add_argument('--checkpoint_folder', default='checkpoint_default', type=str, help='checkpoint folder name')
    parser.add_argument('--model_path', default='', type=str, help='where is the pretrained model?')
    parser.add_argument('--clip_to', default=0.5, type=float, help='clipping for singular values in rank regularizer')
    parser.add_argument('--max_sig', default=1.5, type=float, help='clipping for singular values in rank maximizer')
    parser.add_argument('--rankmin', action='store_true', help='minimize rank?')
    parser.add_argument('--rankmax', action='store_true', help='maximize rank?')
    parser.add_argument('--rank_sched', nargs='+', default=[15, 30, 40], type=int, help='how often to do low rank approx')
    parser.add_argument('--adversarial', action='store_true', help='adversarial training?')
    args = parser.parse_args()

    # helpful vairables
    lr = args.lr
    epochs = args.epochs

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get datasets
    trainloader, testloader, num_channels = get_data_sets(args.dataset)

    config = {
        'epsilon': 4.0 / 255,
        'num_steps': args.attack_steps,
        'step_size': 1.0 / 255,
    }

    # Define model
    if args.model == 'ResNet18':
        basic_net = ResNet18(num_channels=num_channels)
    else:
        print('Model not implemented.')
        return

    # Finetuning setup
    if args.model_path is not '':
        dict = torch.load(args.model_path, map_location='cpu')
        basic_net.load_state_dict(dict['net'])

    net = AttackPGD(basic_net, config, args.adversarial)
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)

    all_losses = []
    test_natural = []
    all_acc = []
    test_robust = []

    params = {'epochs': args.epochs, 'lr': args.lr, 'model': args.model,
              'rankmin': args.rankmin, 'adv': args.adversarial}


    # Train
    print('\n\t==> Training...')
    for epoch in range(epochs):
        sys.stdout.flush()

        # learing rate drop
        adjust_learning_rate(optimizer, epoch, lr, args.lr_schedule, args.lr_factor)
        loss, acc = train(net, trainloader, optimizer)

        # Add current loss to list of losses
        all_losses.append(loss)
        all_acc.append(acc)

        if (epoch + 1) % args.val_period == 0:
            # Tracking test performance
            test_natural_acc, test_robust_acc = test(net, basic_net, testloader, config, params, args.output,
                                                     args.adversarial)
            test_natural.append(test_natural_acc)
            test_robust.append(test_robust_acc)

            print('')
            print('Epoch: ', epoch)
            print('Training loss: ', loss)
            print('Training acc: ', acc)
            print('Test acc: ', test_natural_acc)
            print('Robust acc: ', test_robust_acc)

        if epoch in args.rank_sched and args.rankmin:
            print('(epoch: ', epoch, ', rank min step)')
            clip_network(basic_net, args.clip_to)

        if epoch in args.rank_sched and args.rankmax:
            print('(epoch: ', epoch, ', rank min step)')
            max_rank(basic_net, args.max_sig)

        sys.stdout.flush()

    print('\tDone training.')

    state = {
        'net': basic_net.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict()
    }

    out_dir = os.path.join(args.checkpoint_folder, args.dataset + '_min' + str(args.rankmin) + '_max' + str(args.rankmax) + '_a' + str(args.adversarial))
    print('saving model to: ', out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    torch.save(state, os.path.join(out_dir, 'epoch=' + str(epoch) + '.t7'))

    print("ending  main method in train_net.py")
    print("Time now: ", datetime.datetime.now())
    print("\n_________________________________________________\n")

    return


if __name__ == "__main__":
    main()