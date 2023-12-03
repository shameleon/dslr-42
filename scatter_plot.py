#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scatter_plot.py
    Generate a Scatter plot for two features of a dataset

Usage:
    script [dataset] [feature] [feature]
    example:
    $ python scatter_plot.py ./datasets/dataset_train.csv
    'Astronomy' 'Herbology'
Subject:
    What are the two features that are similar ?
    Answer : 'Astronomy', 'Defense Against the Dark Arts'
"""
__authors__ = ['jmouaike, ebremond']

import argparse
import os


if __name__ == "__main__":
    """Calls dslr/plot_dataset.py with the -s option to scatterplot.
    Executes the command (as string) in a subshell. See os.system doc.
    $ python scatter_plot.py ./datasets/dataset_train.csv 'Astronomy'
     'Herbology'
    """
    description = 'scatter plot for a given feature of a dataset'
    script_path = './dslr/plot_dataset.py'
    parser = argparse.ArgumentParser(prog='scatter_plot.py',
                                     description=description)
    parser.add_argument('filename')
    parser.add_argument('features', nargs=2, type=str)
    args = parser.parse_args()
    print("Scatter plot for", os.path.split(args.filename)[1])
    print('features :', args.features)
    os.system(f'./venv/bin/python {script_path} {args.filename}'
              + f' -s "{args.features[0]}" "{args.features[1]}"')
