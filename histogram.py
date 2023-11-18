import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

""" 
Histogram
Make a script called histogram.[extension] which displays a
histogram answering the next question :
Which Hogwarts course has a homogeneous score distribution
between all four houses?
"""


def plot_histogram(feature='Astrology'):
    try:
        df = pd.read_csv(args.filename)
        df_feat = df.groupby('Hogwarts House')[feature]
        df_feat.plot(kind='hist', alpha=0.4, legend=True)
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
    plot_histogram('Herbology')
