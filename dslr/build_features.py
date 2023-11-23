import numpy as np
import pandas as pd
import config

"""
    select 10 best features for training
"""

def drop_columns(df: pd.DataFrame, drop_na:bool) -> pd.DataFrame:
    """ drop dataframe columns that are not useful for training:
        - non numeric data
        - numeric data that has colinearity
        - not meaningful variables should be included
        drop dataframe rows that contains NaN
    """
    df.drop(df.columns[0], inplace=True, axis = 1)
    df.drop(config.excluded_features, inplace=True, axis=1)
    df.drop(df.columns[5:], inplace=True, axis = 1)
    if drop_na:
        return df.dropna()
    else:
        return df.fillna(0)


def build_dataset():
    """  """
    try:
        df = pd.read_csv(args.csv_file_path)
        training = TrainingDataset(df,
                                   config.learning_rate,
                                   config.epochs)
        print(training)
    except (FileNotFoundError, IsADirectoryError) as e:
        print("File Error :", e)
        sys.exit("No file provided : exit")
    except pd.errors.EmptyDataError as e:
        print("File Content Error :", e)
        sys.exit("Empty File : exit")


if __name__ == "__main__":
    """_summary_
    """
    parser = argparse.ArgumentParser(prog='build_features.py',
                                     description='Process training dataset',
                                     epilog='Enter a valid csv file, please')
    parser.add_argument('csv_file_path')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-b', '--bonus',
                        action='store_true')
    args = parser.parse_args()
    build_dataset()
    sys.exit(0)