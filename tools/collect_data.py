import os
import os.path as osp
import re
import argparse
import numpy as np
from pdb import set_trace

def parse_args():
    parser = argparse.ArgumentParser(description='Experiment summary parser')
    parser.add_argument('--save_dir', default='checkpoints_tune', type=str)
    parser.add_argument('--exp_format', type=str)
    return parser.parse_args()


def read_num(saveDir, exp):
    path = osp.join(saveDir, exp, 'log.txt')
    if not osp.isfile(path):
        return -1
    with open(path, 'r') as file:
        lines = file.read().splitlines()
    bestAcc = -1
    for line in lines[-20:]:
        # set_trace()
        groups = re.match("^Final Test Accuracy of the network on the [0-9]+ test images: ([0-9]+\.[0-9]+)%$", line)
        if groups:
            bestAcc = float(groups[1])

        groups = re.match("^Top 1 acc for best model is ([0-9]+\.[0-9]+)$", line)
        if groups:
            bestAcc = float(groups[1])
    return bestAcc

def main():
    args = parse_args()
    exps_all = os.listdir(args.save_dir)
    exps_select = []
    # set_trace()
    for exp in exps_all:
        if re.match(args.exp_format, exp) is not None:
            exps_select.append(exp)

    exps_select = sorted(exps_select)
    numbers = []
    for exp in exps_select:
        acc = read_num(args.save_dir, exp)
        if acc > 0:
            print("{}: {}".format(exp, acc))
            numbers.append(acc)
        else:
            print("read fail for {}".format(exp))

    if len(numbers) > 0:
        print("mean is {}, std is {} for {}".format(np.mean(np.array(numbers)), np.std(np.array(numbers)), numbers))


if __name__ == "__main__":
    main()
