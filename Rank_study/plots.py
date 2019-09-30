############################################################
#
# plots.py
# plotting functions for rank study
# September 2019
#
############################################################

import matplotlib as mpl
# if os.environ.get('DISPLAY', '') == '':
print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import argparse
import datetime

def main():
    print("\n_________________________________________________\n")
    print("Starting main method in plots.py")
    print("Time now: ", datetime.datetime.now())

    parser = argparse.ArgumentParser(description='Rank study: plot generation')
    parser.add_argument('--model', default='ResNet18', type=str, help='Which model')
    parser.add_argument('--names', nargs='+', default=[''], type=str, help='Which models?')
    parser.add_argument('--output', default='plots', type=str, help='output folder')
    parser.add_argument('--pltnum', default='0', type=str, help='plot number for saving.')
    args = parser.parse_args()
    model = args.model

    # ------------------------------------------------------------------
    # Comparison settings
    # ------------------------------------------------------------------

    runs = [model+'/'+n for n in args.names]

    # Choose parameters to compare (by commenting):

    if model == 'ResNet18':
        params = ['layer1.0',
                  'layer1.1',
                  'layer2.0',
                  'layer2.1',
                  'layer3.0',
                  'layer3.1',
                  'layer4.0',
                  'layer4.1']
    elif model == 'ResNet34':
        params = ['layer1.0',
                  'layer1.1',
                  'layer1.2',
                  'layer2.0',
                  'layer2.1',
                  'layer2.2',
                  'layer2.3',
                  'layer3.0',
                  'layer3.1',
                  'layer3.2',
                  'layer3.3',
                  'layer3.4',
                  'layer3.5',
                  'layer4.0',
                  'layer4.1',
                  'layer4.2']
    else:
        print('Plot generation not yet implemented for model: ', model)
        return

    big_fig, big_ax = plt.subplots(1, 1, figsize=(16, 9))
    for t, out_dir in enumerate(runs):
        for layer, p in enumerate(params):
            i = 0
            for fh in glob.glob(os.path.join(out_dir, '*.npy')):
                if p in fh:
                    print(fh)
                    sig = np.load(fh)
                    effective_rank = np.linalg.norm(sig, ord=1) / np.linalg.norm(sig, ord=2)
                    if p == 'layer1.0' and i == 0:
                        mylabel = args.names[t]
                    else:
                        mylabel = None
                    big_ax.plot(2*(layer+i*0.5), effective_rank, 'kbrgycm'[t]+'o^*+0^*+'[t], label=mylabel, markersize=12)
                    i += 1

    # ------------------------------------------------------------------
    # Plotting stuff
    # ------------------------------------------------------------------
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    big_ax.set_xlabel("Filter index", fontsize=28)
    big_ax.set_ylabel("Effective Rank", fontsize=28)
    big_ax.legend(fontsize=20)
    plt.subplots_adjust(right=0.85)
    big_fig.savefig(os.path.join(args.output, out_dir[:8] + args.pltnum + '.pdf'))

    print('done plotting.')
    print("ending  main method in plots.py")
    print("Time now: ", datetime.datetime.now())
    print("\n_________________________________________________\n")
    return

if __name__ == '__main__':
    main()