import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import config
from utils import print_out as pout

def dataset_plotter(func):
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
            raise SystemExit(1)
        except pd.errors.EmptyDataError as e:
            print("File Content Error :", e)
            raise SystemExit(1)
        except ValueError as e:
            print("Argument Error :", e)
            raise SystemExit(1)
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


def plot_boxplot(df: pd.DataFrame, target_label: str, feature: str):
    sns.boxplot(data=df, x=target_label, y=feature)
    plt.title(feature)
    return None


def plot_scatter(df: pd.DataFrame, target_label: str,
                feature1: str, feature2=str):
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


def plot_pairplotmatrix(df: pd.DataFrame, target_label: str,
                        features: list):
    sns.pairplot(df,
                 x_vars=features,
                 y_vars=features,
                 hue=target_label,
                 corner=True)
    sns.set(rc={'font.size' : 6, 'axes.labelsize': 6})
    plt.savefig('./reports/pair_plot_matrix.png')


def plot_joinplot(df: pd.DataFrame, target_label: str,
                feature1: str, feature2=str):
    sns.jointplot(data=df, x=feature1, y=feature2, hue=target_label)


@dataset_plotter
def plot_selected():
    df = pd.read_csv(args.filepath)
    df.drop(['Index'], inplace=True, axis=1)
    df_features = df.select_dtypes(include=np.number).columns
    if args.histogram:
        plot_histogram(df, config.target_label, args.histogram[0])
    elif args.boxplot:
        plot_boxplot(df, config.target_label, args.boxplot[0])
    elif args.scatter:
        plot_scatter(df, config.target_label,
                     args.scatter[0], args.scatter[1])
    elif args.joinplot:
        plot_joinplot(df, config.target_label,
                      args.joinplot[0], args.joinplot[1])
    elif args.pairplotmatrix:
        plot_pairplotmatrix(df, config.target_label,
                            df_features.to_list())
    else:
        print("Data_features :\n", df_features.to_list())
    return None

        
if __name__ == "__main__":
    """ -h for help
    Positional argument : filepath
    Optional Arguments : -h for help, -i, -s, -b, -j"""
    parser = argparse.ArgumentParser(prog='plot_dataset.py',
                                     description='many available plots',
                                     epilog='end')
    parser.add_argument('filepath')
    parser.add_argument('-i', '--histogram', nargs=1, type=str)
    parser.add_argument('-s', '--scatter', nargs=2, type=str)
    parser.add_argument('-p', '--pairplotmatrix', action='store_true')
    parser.add_argument('-b', '--boxplot', nargs=1, type=str)
    parser.add_argument('-j', '--joinplot', nargs=2, type=str)
    args = parser.parse_args()
    print(vars(args))
    plot_selected()
