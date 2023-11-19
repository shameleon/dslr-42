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


def sns_scatter(feature1='Astronomy', feature2='Herbology'):
    """_summary_
        scatter plot for two features with seaborn module
        legend : target categories
    Args:
        feature1 (str, optional): _description_. Defaults to 'Astronomy'.
        feature2 (str, optional): _description_. Defaults to 'Herbology'.
    """
    target = "Hogwarts House"
    try:
        df = pd.read_csv(args.filename)
        house = df.keys()[1]
        plt.figure(figsize=(8,8))
        sns.scatterplot(data=df, 
                        x=feature1,
                        y=feature2,
                        hue=target,
                        legend='auto')
        plt.title("Scatter plot by house")
        plt.legend(loc='upper center',
                   bbox_to_anchor=(0.9, 1.15),
                   ncol=2,
                   title=target)
        plt.show()
    except (FileNotFoundError, IsADirectoryError) as e:
        print("File Error :", e)
    except pd.errors.EmptyDataError as e:
        print("File Content Error :", e)


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser(prog='scatter_plot.[ext]',
                                     description='scatter plot for 2 features of a dataset',
                                     epilog='verbose mode for options')
    parser.add_argument('filename')
    args = parser.parse_args()
    sns_scatter()
    sns_scatter('Astronomy', 'Defense Against the Dark Arts')
