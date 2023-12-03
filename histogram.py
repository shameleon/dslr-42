#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
histogram.py
    Generate a histogram plot for a feature of a dataset

Usage:
    script [dataset] [feature]
    example:
    $ python histogram.py ./datasets/dataset_train.csv 'Charms'

Subject:
    The histogram answering the next question :
    Which Hogwarts course has a homogeneous score distribution
    between all four houses?
    Answer:
        'Arithmancy', 'Care of Magical Creatures'
"""
__authors__ = ['jmouaike, ebremond']

import os

import argparse


if __name__ == "__main__":
    """Calls dslr/plot_dataset.py script with the -i option for histogram
    Executes the command (as string) in a subshell. See os.system doc.
    $ python histogram.py ./datasets/dataset_train.csv 'Charms'
    """
    description = 'histogram for a given feature of a dataset'
    script_path = './dslr/plot_dataset.py'
    parser = argparse.ArgumentParser(prog='histogram.py',
                                     description=description)
    parser.add_argument('filename')
    parser.add_argument('feature', nargs=1, type=str)
    args = parser.parse_args()
    print("Histogram plot for", os.path.split(args.filename)[1])
    print('feature :', args.feature[0])
    os.system(f'./venv/bin/python {script_path} {args.filename}' +
              f' -i "{args.feature[0]}"')
