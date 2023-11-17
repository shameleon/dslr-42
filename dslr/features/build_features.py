import numpy as np
import pandas as pd
from ..stat_utils import standardize

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
    excluded_features = ["Arithmancy",
                         "Defense Against the Dark Arts",
                         "Care of Magical Creatures"]
    df.drop(excluded_features, inplace=True, axis=1)
    df.drop(df.columns[5:], inplace=True, axis = 1)
    if drop_na:
        return df.dropna()
    else:
        return df.fillna(0)


def main():
    model_dir = './logistic_reg_model/'
    df = pd.read_csv(f'./datasets/dataset_train.csv')
    df2 = drop_columns(df, False)


if __name__ == "__main__":
    """ train model from file """
    main()