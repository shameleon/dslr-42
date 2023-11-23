import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import config

def data_plotter(func):
    """
    This is a decorator to get exceptions while 
    reading dataset file then plotting dataset features
    """
    def plot_wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
            plt.show()
        except (FileNotFoundError, IsADirectoryError) as e:
            print("File Error :", e)
        except pd.errors.EmptyDataError as e:
            print("File Content Error :", e)
        return None
    return plot_wrapper

"""
Subfunctions used by plot_selected()
"""
def plot_histogram(df: pd.DataFrame, target_label: str, feature: str):
    df_feat = df.groupby(target_label)[feature]
    df_feat.plot(kind='hist', alpha=0.4, legend=True)
    plt.title(feature)
    return None


def sns_boxplot(df: pd.DataFrame, target_label: str, feature: str):
    sns.boxplot(data=df, x=target_label, y=feature)
    plt.title(feature)
    return None


def sns_scatter(df: pd.DataFrame, target_label: str, feature1: str, feature2=str):
    plt.figure(figsize=(8,8))
    sns.scatterplot(data=df, 
                    x=feature1,
                    y=feature2,
                    hue=target_label,
                    legend='auto')
    plt.title("Scatter plot by house")
    plt.legend(loc='upper center',
                bbox_to_anchor=(0.9, 1.15),
                ncol=2,
                title=target_label)
    return None


@data_plotter
def plot_selected():
    df = pd.read_csv(args.filename)
    if args.histogram:
        plot_histogram(config.target_label, args.histogram)
    return None

        
if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser(prog='all_plots.py',
                                     description='plot histogram',
                                     epilog='')
    parser.add_argument('filename')
    parser.add_argument('-i', '--histogram',
                        action='store_true')
    parser.add_argument('-s', '--scatter',
                        action='store_true')
    parser.add_argument('-p', '--pairplotmatrix',
                        action='store_true')
    args = parser.parse_args()
    plot_selected()