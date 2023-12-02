import argparse
import os

"""logreg_predict.py:

Program to test a logistic regression model
Comparing predicted outputs to real outputs
"""


if __name__ == "__main__":
    """
    Train model from dataset.

    Arguments (2)
        - filepath to .csv file of dataset to train
        - weigths file path to .csv file of model weights

    Prerequisites :
        - set up the ./dslr/config.py file
        - model must have been trained
    """
    script_path = './dslr/predict_dataset.py'
    parser = argparse.ArgumentParser(prog='logreg_predict.py',
                                     description='Testing dataset',
                                     epilog='verbose mode for options')
    parser.add_argument('filepath', type=str)
    parser.add_argument('weights', type=str)
    args = parser.parse_args()
    print(f'Testing dataset : {args.filepath}')
    print(f'Model weights : {args.weights}')
    os.system(f'./venv/bin/python {script_path}'
              + f' {args.filepath} {args.weights}')
