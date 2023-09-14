import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def standardize(arr:np.ndarray):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std


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
    # normalize
    df2['Best Hand'].replace({"Left":0, "Right":1}, inplace=True)
    df2['First Initial'] = df2['First Name'].str[0]
    df2['First Initial'] = [ ord(x) - 64 for x in df2['First Initial'] ]
    #df2['First Initial'].map(lambda x: ord(x) - 64 if x.isalpha() else x, line)
    #df2['First Initial'] = df2['First Name'].replace('[^A-Z]', '', regex=True).astype('int64')
    df2['Last Initial'] = df2['Last Name'].str[0]
    df2['Last Initial'] = [ ord(x) - 64 for x in df2['Last Initial'] ]
    df2.drop(['First Name', 'Last Name'], inplace=True, axis=1)
    df2['Year'] = pd.DatetimeIndex(df2['Birthday']).year
    df2['Month'] = pd.DatetimeIndex(df2['Birthday']).month
    df2['Day'] = pd.DatetimeIndex(df2['Birthday']).day
    df2.drop(['Birthday'], inplace=True, axis=1)
    print(df2.head())
    #pd.merge(df1, df2, left_index=True, right_index=True)
    #sns.swarmplot(data=df2, x='Month', y='Last Initial', hue='Hogwarts House', legend=True)
    sns.histplot(data=df2, x='Best Hand', hue='Hogwarts House', legend=True)
    # g = sns.catplot(
    # data=df2, x="First Initial", y="Year", hue="Hogwarts House", col="Best Hand",
    # capsize=.2, kind="point", height=6, aspect=.75,)
    # g.despine(left=True)
    plt.show()
    
    # df_inexact.drop(df.columns[6:18], inplace=True, axis = 1)
    # df_inexact.sort_values(['First Name', 'Best Hand'], inplace=True)
    # df_inexact.to_csv(f'{model_dir}incorrect_prediction_for_trainset1600.csv')

    #sns.histplot(data=df, x="Accurate pred.'", color="skyblue", kde=True, hue="Hogwarts House")
    #sns.histplot(data=df, x="Herbology", color="skyblue", kde=True, hue="Accurate pred.")
          

if __name__ == "__main__":
    """ train model from file """
    main()