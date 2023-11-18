import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
V.2.2 Scatter plot
Make a script called scatter_plot.[extension] which displays
a scatter plot answering the next question :
What are the two features that are similar ?
answer : sns_scatter('Astronomy', 'Defense Against the Dark Arts')
"""


def plot_scatter(feature1='Astronomy', feature2='Herbology'):
    try:
        df = pd.read_csv(args.filename)
        house = df.keys()[1]
        plt.scatter(feature1, feature2, marker='.', alpha=0.3, data=df)
        plt.show()
    except (FileNotFoundError, IsADirectoryError) as e:
        print("File Error :", e)
    except pd.errors.EmptyDataError as e:
        print("File Content Error :", e)

def sns_scatter(feature1='Astronomy', feature2='Herbology'):
    try:
        df = pd.read_csv(args.filename)
        house = df.keys()[1]
        sns.scatterplot(data=df, 
                        x=feature1,
                        y=feature2,
                        hue="Hogwarts House",
                        legend='auto'
                        )
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
    sns_scatter()
    sns_scatter('Astronomy', 'Defense Against the Dark Arts')
