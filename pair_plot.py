import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
