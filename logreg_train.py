import argparse
import os

"""logreg_train.py:

Program to train a logistic regression model
based on gradient descent algorithm to minimize the error

Usage example
$ python logreg_train.py ./datasets/dataset_train.csv
"""


if __name__ == "__main__":
    """
    Train model from dataset.

    Argument (1): filepath to .csv file of dataset to train

    Prerequisite : set up the ./dslr/config.py file
        to select the target, important features,
        learning rate and epochs
    """
    script_path = './dslr/training_dataset.py'
    parser = argparse.ArgumentParser(prog='logreg_train.py',
                                     description='Training dataset',
                                     epilog='verbose mode for options')
    parser.add_argument('filepath')
    args = parser.parse_args()
    os.system(f'./venv/bin/python {script_path} {args.filepath}')
