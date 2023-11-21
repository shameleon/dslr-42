import argparse
import numpy as np
import os
import pandas as pd
import sys
from MultinomialTrainClass import LogRegTrain

""" """

__author__ = "jmouaike"


class TrainingDataset:
    """_summary_
    """
    def __init__(self, filepath):
        self.filepath = filepath
        if self.read_dataset():
            train_logistic_regression_model()

    def read_dataset(self) -> bool:
        """  """
        try:
            df = pd.read_csv(args.filename)
            return True
        except (FileNotFoundError, IsADirectoryError) as e:
            print("File Error :", e)
            sys.exit("No file exit")
        except pd.errors.EmptyDataError as e:
            print("File Content Error :", e)
            sys.exit("Empty File exit")

    def train_logistic_regression_model(self):


    def save_weights(self):
        model_dir = './logistic_reg_model/'
        dest_file = 'gradient_descent_weights.csv'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.df_weights.to_csv(f'{model_dir + dest_file}')


def drop_useless(df: pd.DataFrame) -> pd.DataFrame:
    """ drop dataframe columns that are not useful for training:
        - non numeric data
        - numeric data that has colinearity
        - not meaningful variables should be included
        drop dataframe rows that contains NaN
    """
    df.drop(df.columns[2:6], inplace=True, axis = 1)
    excluded_features = ["Arithmancy",
                         "Defense Against the Dark Arts",
                         "Care of Magical Creatures"]
    df.drop(excluded_features, inplace=True, axis=1)
    return df.dropna()


def train_dataset():
    if not read_dataset():
        return None


if __name__ == "__main__":
    """_summary_
    """
    parser = argparse.ArgumentParser(prog='logreg_train.[ext]',
                                     description='Training a dataset',
                                     epilog='Enter a valid csv file, please')
    parser.add_argument('filepath')
    parser.add_argument('-v', '--verbose',
                        action='store_true') 
    parser.add_argument('-b', '--bonus',
                        action='store_true')
    args = parser.parse_args()
    training = TrainingDataset(args.filepath)
    sys.exit(0)

