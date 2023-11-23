import argparse
import pandas as pd
from DescriberClass import Describe

"""describe.py:
    This program reproduces the pandas.decribe() function.

Parameter  a dataset file name as parameter.
options:
    --verbose or -v for more info
    --bonnus or -b for bonus, additonal metrics to describe

An instance of Describe Class is created
with agg_describe() method to display description of features
"""

__author__ = "jmouaike"


def dataset_describer():
    """  """
    if args.verbose:
        print("File     :", args.filename)
        print("Verbose : compares to pandas.describe()")
    try:
        df = pd.read_csv(args.filename)
        if args.bonus:
            print("Bonus")
        dataset_descriptor = Describe(df)
        dataset_descriptor.agg_describe(args.bonus)
    except (FileNotFoundError, IsADirectoryError) as e:
        print("File Error :", e)
    except pd.errors.EmptyDataError as e:
        print("File Content Error :", e)


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
