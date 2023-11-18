import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

""" 
Pair plot matrix
V.2.3 Pair plot
Make a script called pair_plot.[extension] which displays a pair plot or scatter plot
matrix (according to the library that you are using).
From this visualization, what features are you going to use for your logistic regression?
7
"""


def plot_pairs():
    try:
        df = pd.read_csv(args.filename)
        house = df.keys()[1]
        plt.scatter(feature1, feature2, marker='.', alpha=0.3, data=df)
        plt.show()
    except (FileNotFoundError, IsADirectoryError) as e:
        print("File Error :", e)
    except pd.errors.EmptyDataError as e:
        print("File Content Error :", e)


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser(prog='histogram.[ext]',
                                     description='histogram for a given feature of a dataset',
                                     epilog='verbose mode for options')
    parser.add_argument('filename')
    args = parser.parse_args()
    plot_pairs()
