import os
import csv
import numpy as np


def save_output(out_dir, dataset, model, norm_bias, weight_decay_coef, train_loss, test_loss, acc, weight_norm):

    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, 'table.csv')
    fieldnames = ['dataset', 'model', 'norm_bias', 'weight_decay_coef', 'train_loss', 'test_loss', 'acc', 'weight_norm']

    # Read or write header
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = [line for line in reader][0]
    except:
        with open(fname, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()

    # Add row for this experiment
    with open(fname, 'a') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        writer.writerow({'dataset': dataset,
                         'model': model,
                         'norm_bias': norm_bias,
                         'weight_decay_coef': weight_decay_coef,
                         'train_loss': train_loss,
                         'test_loss': test_loss,
                         'acc': acc,
                         'weight_norm': weight_norm})
    print('\nResults saved to '+fname+'.')
