import argparse
import os

"""logreg_predict.py:

Program to test a logistic regression model
comparing predicted outputs to real outputs

"""
    

if __name__ == "__main__":
    """
    Train model from dataset.

    Argument : filepath to .csv file of dataset to train

    Prerequisite : set up the ./dslr/config.py file 

    """
    script_path = './dslr/training_dataset.py'
    parser = argparse.ArgumentParser(prog='logreg_predict.py',
                                     description='Testing dataset',
                                     epilog='verbose mode for options')
    parser.add_argument('testfilepath')
    parser.add_argument('weights')
    args = parser.parse_args()
    os.system(f'./venv/bin/python {script_path} {args.testfilepath} {args.weights}')
