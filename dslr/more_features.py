#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


"""dslr/more_features
    in addition to the 10 best features,
    exploring additonal features for training:

"""


def drop_columns(df: pd.DataFrame, drop_na: bool) -> pd.DataFrame:
    """ drop dataframe columns that are not useful for training:
        - non numeric data
        - numeric data that has colinearity
        - not meaningful variables should be included
        drop dataframe rows that contains NaN
    """
    df.drop(df.columns[0], inplace=True, axis=1)
    excluded_features = ["Arithmancy",
                         "Defense Against the Dark Arts",
                         "Care of Magical Creatures"]
    df.drop(excluded_features, inplace=True, axis=1)
    df.drop(df.columns[5:], inplace=True, axis=1)
    if drop_na:
        return df.dropna()
    else:
        return df.fillna(0)


def main():
    df = pd.read_csv('../datasets/dataset_train.csv')
    df2 = drop_columns(df, False)
    df2['Best Hand'].replace({"Left": 0, "Right": 1}, inplace=True)
    df2['First Initial'] = df2['First Name'].str[0]
    df2['First Initial'] = [ord(x) - 64 for x in df2['First Initial']]
    df2['Last Initial'] = df2['Last Name'].str[0]
    df2['Last Initial'] = [ord(x) - 64 for x in df2['Last Initial']]
    df2.drop(['First Name', 'Last Name'], inplace=True, axis=1)
    df2['Year'] = pd.DatetimeIndex(df2['Birthday']).year
    df2['Month'] = pd.DatetimeIndex(df2['Birthday']).month
    df2['Day'] = pd.DatetimeIndex(df2['Birthday']).day
    df2.drop(['Birthday'], inplace=True, axis=1)
    print(df2.head())
    sns.histplot(data=df2, x='Best Hand', hue='Hogwarts House', legend=True)
    plt.show()


if __name__ == "__main__":
    """ train model from file """
    main()
