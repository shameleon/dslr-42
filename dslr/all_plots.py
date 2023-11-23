import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config

def overplot(func):
    """
    This is a decorator to get exceptions while plotting dataset
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


@overplot
def plot_histogram(target_label: str, feature='Astrology'):
    df = pd.read_csv(args.filename)
    df_feat = df.groupby(target_label)[feature]
    df_feat.plot(kind='hist', alpha=0.4, legend=True)
    plt.title(feature)


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser(prog='all_plots.py',
                                     description='plot histogram',
                                     epilog='verbose mode for options')
    parser.add_argument('filename')
    args = parser.parse_args()
    plot_histogram(config.target_label, 'Herbology')