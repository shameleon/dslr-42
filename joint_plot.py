import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

""" 
Joint plot : Draw a plot of two variables with bivariate and univariate graphs.
https://seaborn.pydata.org/generated/seaborn.jointplot.html
"""


def joint_plot(feature1="Astrology", feature2="Herbology"):
    """_summary_

    Args:
        feature1 (str): _description_
        feature2 (str): _description_
    """
    target = "Hogwarts House"
    try:
        df = pd.read_csv(args.filename)
        house = df.keys()[1]
        sns.jointplot(data=df, x=feature1, y=feature2, hue=target)
        plt.show()
    except (FileNotFoundError, IsADirectoryError) as e:
        print("File Error :", e)
    except pd.errors.EmptyDataError as e:
        print("File Content Error :", e)


if __name__ == "__main__":
    """_summary_ 
    """
    parser = argparse.ArgumentParser(prog='joint_plot.[ext]',
                                     description='joint plot for two given features of a dataset',
                                     epilog='verbose mode for options')
    parser.add_argument('filename')
    args = parser.parse_args()
    joint_plot()
