import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


"""
    boxplot with seaborn module
"""

def sns_boxplot(feature='Astronomy'):
    try:
        df = pd.read_csv(args.filename)
        house = df.keys()[1]
        sns.boxplot(data=df, 
                    x="Hogwarts House",
                    y=feature
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
    sns_boxplot()
