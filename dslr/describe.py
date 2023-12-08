#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dslr/describe.py:
    This program reproduces the pandas.decribe() function.

Parameter
    a dataset file name as parameter.
options
    --verbose or -v for more info
    --bonus or -b for bonus, additonal metrics to describe

An instance of Describe Class is created
with agg_describe() method to display description of features
"""
__authors__ = ['jmouaike, ebremond']

import argparse
import pandas as pd

from DescriberClass import Describe
from utils import print_out as po


def dataset_describer():
    """  """
    if args.verbose:
        print("File     :", args.filename)
        print("Verbose mode")
    try:
        df = pd.read_csv(args.filename)
        if args.bonus:
            po.as_status("describe.py bonus ")
        dataset_descriptor = Describe(df)
        dataset_descriptor.agg_describe(args.bonus)
    except (FileNotFoundError, IsADirectoryError) as e:
        po.as_error2("File Error :", e)
    except pd.errors.EmptyDataError as e:
        po.as_error2("File Content Error :", e)


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser(prog='describe.[ext]',
                                     description='Describe a dataset',
                                     epilog='verbose mode for options')
    parser.add_argument('filename')
    parser.add_argument('-b', '--bonus',
                        action='store_true')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    args = parser.parse_args()
    dataset_describer()


"""
Numerical column stats:

[+]     No. of Values = count
[+]     Mean
[+]     Std
[+]     Min
[+]     First quartile
[+]     Median
[+]     Third quartile
[+]     Max
[+]     No. of NaN

bonus :
"""
