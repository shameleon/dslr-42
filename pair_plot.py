import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

""" 
Pair plot matrix : Plot pairwise relationships in a dataset.
V.2.3 Pair plot
Make a script called pair_plot.[extension] which displays a pair plot or scatter plot
matrix (according to the library that you are using).
From this visualization, what features are you going
to use for your logistic regression?
7
"""


def plot_pair_matrix(save_plot=False):
    target="Hogwarts House"
    remove_list = ['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    try:
        df = pd.read_csv(args.filename)
        df = df.drop(remove_list, axis=1)
        features = df.keys()[1:].to_list()
        sns.pairplot(df,
                     x_vars=features,
                     y_vars=features,
                     hue=target,
                     corner=True
                    )
        if save_plot:
            plt.savefig('./reports/pair_plot_matrix.png')
        plt.show()
    except (FileNotFoundError, IsADirectoryError) as e:
        print("File Error :", e)
        print("Please, provide a valid .csv file as argument")
    except pd.errors.EmptyDataError as e:
        print("File Content Error :", e)


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser(prog='pair_plot[ext]',
                                     description='matrix pairplot for a dataset',
                                     epilog='verbose mode for options')
    parser.add_argument('filename')
    args = parser.parse_args()
    plot_pair_matrix(True)
