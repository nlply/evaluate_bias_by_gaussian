import json
import argparse
import csv
from random import sample
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_rate', type=int, default=0.8)
    parser.add_argument('--input', type=str, default='../data/')
    parser.add_argument('--output', type=str, default='../data/')
    args = parser.parse_args()

    return args


def sample_data(file_name, rate):
    file_name = file_name + '.json'
    with open(file_name) as f:
        input = json.load(f)
        input_len = len(input)
        sample_num = input_len * rate
        return sample(input, math.ceil(sample_num))


def main(args):
    file_names = ['paralled_cp', 'paralled_ss']
    for file_name in file_names:
        data = sample_data(args.input + file_name, args.sample_rate)
        with open(args.output + file_name + '_' + str(args.sample_rate) + '.json', 'w') as fw:
            json.dump(data, fw, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
