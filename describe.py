#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
describe.py
    Describes the data of a dataset file,
    executes ./dlsr/describe.py script
    takes one argument 'filename' .csv file

    Usage:
    python describe.py ./datasets/dataset_train.py
"""
__authors__ = ['jmouaike, ebremond']

import argparse
import os


if __name__ == "__main__":
    """ """
    script_path = './dslr/describe.py'
    parser = argparse.ArgumentParser(prog='describe.[ext]',
                                     description='describe data of a dataset',
                                     epilog='verbose mode for options')
    parser.add_argument('filename')
    parser.add_argument('-b', '--bonus', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    option = ' ' + ' -b' * args.bonus + ' -v' * args.verbose
    os.system(f'./venv/bin/python {script_path} {args.filename} {option}')
