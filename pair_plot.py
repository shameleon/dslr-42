#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pair_plot.py
    Pair plot matrix : Plot pairwise relationships in a dataset.

Usage:
    script [dataset]
     $ python pairplot_plot.py ./datasets/dataset_train.csv

Subject:
    From this visualization, what features are you going
    to use for your logistic regression?
    Answer : see dslr/config.py for excluded features
"""
__authors__ = ['jmouaike, ebremond']

import argparse
import os


if __name__ == "__main__":
    """Calls dslr/plot_dataset.py with the -p option to obtain
    a pairplot triangular matrix of the dataset.
    Executes the command (as string) in a subshell. See os.system doc.
    $ python pair_plot.py ./datasets/dataset_train.csv
    """
    description = 'Pairplot matrix for all features of a dataset'
    script_path = './dslr/plot_dataset.py'
    parser = argparse.ArgumentParser(prog='pair_plot.py',
                                     description=description)
    parser.add_argument('filename')
    args = parser.parse_args()
    os.system(f'./venv/bin/python {script_path} {args.filename} -p')
