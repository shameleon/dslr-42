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
    def __init__(self, df: pd.DataFrame, learning=0.1, epochs=1000):
        """_summary_

        Args:
            df (pd.Dataframe): _description_ df[0] are indexes,
                                             df[1] are real outputs
            learning (float, optional): _description_. Defaults to 0.1.
            epochs (int, optional): _description_. Defaults to 1000.
        """
        self.df = df
        self.learning_rate = learning
        self.epochs = epochs
        self.excluded_features = ["Arithmancy",
                                  "Defense Against the Dark Arts",
                                  "Care of Magical Creatures"]
        self.model_dir = './logistic_reg_model/'
        self.dest_file = 'gradient_descent_weights.csv'
        self.train_logistic_regression_model()

    def train_logistic_regression_model(self):
        """_summary_

        dataframe features are selected. 
        Outputs are kept    
        dataframe means and std are saved to csv.file
        issue : use mean and std from utlis
        an instance of LogRegTrain class is constructed
        and takes 2 parameters :
            - features data (df_x_train) NOT STANDARDIZED
            - corresponding real outputs (df_class)
        LogRegTrain class will standardize and train model,
        returning a dataframe for weights
        """
        df_train = self.drop_useless_features()
        target = df_train.columns[1]
        df_x_train = df_train[df_train.columns[2:]]
        df_normalized = pd.DataFrame({'mean_x': df_x_train.mean(axis=0),
                                       'std_x': df_x_train.std(axis=0)}).T
        df_normalized.to_csv(f'{self.model_dir}normalization.csv')
        df_class = df_train[target]
        logreg_model = LogRegTrain(df_x_train, df_class)
        self.df_weights = logreg_model.train(self.learning_rate, self.epochs)
        self.save_weights()
        df_pred_proba = logreg_model.get_predict_proba()
        df_pred_proba.to_csv(f'{self.model_dir}prediction_for_trainset1333.csv')
        print("Prediction accurate at", 100 * np.mean(df_pred_proba['Accurate pred.']), "%.") 

    def drop_useless_features(self) -> pd.DataFrame:
        """Drop dataframe columns that are not useful for training:
            - non numeric data
            - numeric data that has colinearity
            - not meaningful variables should be included
            drop dataframe rows that contains NaN
        """
        df = self.df
        df.drop(df.columns[2:6], inplace=True, axis = 1)
        df.drop(self.excluded_features, inplace=True, axis=1)
        return df.dropna()

    def save_weights(self):
        model_dir = self.model_dir
        dest_file = self.dest_file
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.df_weights.to_csv(f'{model_dir + dest_file}')


def train_dataset():
    """  """
    try:
        df = pd.read_csv(args.csv_file_path)
        training = TrainingDataset(df)
    except (FileNotFoundError, IsADirectoryError) as e:
        print("File Error :", e)
        sys.exit("No file provided : exit")
    except pd.errors.EmptyDataError as e:
        print("File Content Error :", e)
        sys.exit("Empty File : exit")


if __name__ == "__main__":
    """_summary_
    """
    parser = argparse.ArgumentParser(prog='logreg_train.[ext]',
                                     description='Training a dataset',
                                     epilog='Enter a valid csv file, please')
    parser.add_argument('csv_file_path')
    parser.add_argument('-v', '--verbose',
                        action='store_true') 
    parser.add_argument('-b', '--bonus',
                        action='store_true')
    args = parser.parse_args()
    train_dataset()
    sys.exit(0)

