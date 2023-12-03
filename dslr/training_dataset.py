#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dslr/training_dataset.py"""
__authors__ = ['jmouaike, ebremond']

import os
import sys

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config
from MultinomialTrainClass import LogRegTrain
from utils import describe_stats as dum
from utils import print_out as po


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
        Setting up the config.py file is necessary
        """
        self.df = df
        self.learning_rate = learning
        self.epochs = epochs
        self.excluded_features = config.excluded_features
        self.model_dir = config.model_dir
        self.dest_file = config.gradient_descent_weights
        po.as_title("Training dataset")
        po.as_title2(f'Sample size : {df.shape[0]}')
        self.train_logistic_regression_model()

    def train_logistic_regression_model(self):
        """_summary_

        df dataframe features are selected, so that
        df_train contains only target variable column
        and meaningful data as unstandardized data.

        The means and std of dataset features are saved to csv file.

        issue : use mean and std from utlis, in
                save_mean_and_std method()

        An instance of LogRegTrain class is constructed
        and takes 2 parameters :
            - features data (df_x_train) NOT STANDARDIZED
            - corresponding real outputs (df_class)
        LogRegTrain class will standardize and train model,
        returning a dataframe for weights
        """
        df_train = self.drop_useless_features()
        target = config.target_label
        df_x_train = df_train[df_train.columns[2:]]
        df_y_class = df_train[target]
        self.save_mean_and_std(df_x_train)
        logreg_model = LogRegTrain(df_x_train, df_y_class)
        self.df_weights = logreg_model.train(self.learning_rate, self.epochs)
        self.save_weights()
        self._losses = logreg_model.get_losses()
        df_pred_proba = logreg_model.get_predict_proba()
        df_pred_proba.to_csv(f'{self.model_dir}prediction_trainset1333.csv')
        accuracy = df_pred_proba['Accurate pred.'].value_counts(1)[1]
        po.as_result(f'Prediction accuracy = {100 * accuracy:.4f} %')

    def drop_useless_features(self) -> pd.DataFrame:
        """Drop dataframe columns that are not useful for training:
            - non numeric data
            - numeric data that has colinearity
            - not meaningful variables should be included
            drop dataframe rows that contains NaN
        """
        df = self.df
        df.drop(df.columns[2:6], inplace=True, axis=1)
        df.drop(self.excluded_features, inplace=True, axis=1)
        return df.dropna()

    def save_mean_and_std(self, df_x_train: pd.DataFrame):
        """ Saves standardization parameters
        Writes means and std's of the dataset to a file
        before standardization that occurs at
        LogRegTrain instance construction.

        df_stats = pd.DataFrame({'mean_x': df_x_train.mean(axis=0),
                                 'std_x': df_x_train.std(axis=0)}).T
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        dest_file = config.standardization_params
        df_means = df_x_train.agg(lambda feat: dum.mean(feat), axis=0)
        df_stds = df_x_train.agg(lambda feat: dum.std(feat), axis=0)
        df_stats = pd.DataFrame({'mean_x': df_means,
                                 'std_x': df_stds}).T
        df_stats.to_csv(f'{self.model_dir + dest_file}')

    def save_weights(self):
        model_dir = self.model_dir
        dest_file = self.dest_file
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.df_weights.to_csv(f'{model_dir + dest_file}')
        print(f'Model Weights to {model_dir + dest_file}')

    def plot_losses_and_weights(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
        fig.suptitle('Dataset training', c='blue', fontsize=16)
        fig.subplots_adjust(hspace=0.125, wspace=0.5)
        sns.lineplot(ax=axes[0], data=self._losses,
                     linestyle='solid', alpha=0.6)
        axes[0].set_title(f'Loss for each {config.target_label}')
        sns.heatmap(ax=axes[1], data=self.df_weights, cmap="YlGnBu")
        axes[1].set_title('Weights')
        plt.show()

    def __str__(self):
        return 'Model ready !'


def train_dataset():
    """  """
    try:
        df = pd.read_csv(args.csv_file_path)
        training = TrainingDataset(df,
                                   config.learning_rate,
                                   config.epochs)
        po.as_check(training)
        if args.verbose:
            training.plot_losses_and_weights()
    except (FileNotFoundError, IsADirectoryError) as e:
        po.as_error2("File Error :", e)
        sys.exit("No file provided : exit")
    except pd.errors.EmptyDataError as e:
        po.as_error2("File Content Error :", e)
        sys.exit("Empty File : exit")


if __name__ == "__main__":
    """_summary_
    -v option for plotting losses and weights)
    """
    parser = argparse.ArgumentParser(prog='training_dataset.py',
                                     description='Training a dataset',
                                     epilog='Enter a valid csv file, please')
    parser.add_argument('csv_file_path')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-b', '--bonus', action='store_true')
    args = parser.parse_args()
    train_dataset()
    sys.exit(0)
