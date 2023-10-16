import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
"""

__author__ = "jmouaike"

def sigmoid(arr:np.ndarray):
    return 1 / (1 + np.exp(-arr))

def predict_proba(x,
                  y,
                  weights,
                  outcomes) -> pd.DataFrame:
    """
    input is the dataset, normalized and containing data usefull for model
    returns : a dataframe containing the probability for each outcome
    and the final predicted outcome
    """
    ones = np.ones((len(x), 1), dtype=float)
    x_test = np.concatenate((ones, x), axis=1)
    print(x_test.shape, weights.shape)
    z = np.dot(x_test, weights)
    # odds = np.log(z)
    h = sigmoid(z)
    df_pred_proba = pd.DataFrame(h, columns=outcomes)
    df_pred_proba['Predicted outcome'] = df_pred_proba.idxmax(axis=1)
    df_pred_proba['Real outcome'] = y.tolist()
    df_pred_proba['Accurate pred.'] = np.where(df_pred_proba['Predicted outcome'] 
                                        == df_pred_proba['Real outcome'], 1, 0)
    return df_pred_proba

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
    df.drop(df.columns[2:6], inplace=True, axis = 1)
    excluded_features = ["Arithmancy",
                         "Defense Against the Dark Arts",
                         "Care of Magical Creatures"]
    df.drop(excluded_features, inplace=True, axis=1)
    if drop_na:
        return df.dropna()
    else:
        return df.fillna(0)

def main():
    model_dir = './logistic_reg_model/'
    df = pd.read_csv(f'./datasets/dataset_train.csv')
    df_train = drop_columns(df, False)
    target = df_train.columns[1]
    df_x_train = df_train[df_train.columns[2:]]
    df_x_train_std = df_x_train.agg(lambda feature: standardize(feature))
    df_y_train = df_train[target]
    df_y_train.unique().tolist()
    model_weights = pd.read_csv(f'./logistic_reg_model/gradient_descent_weights.csv')
    model_weights.drop(columns="Unnamed: 0", inplace=True)
    weights = np.array(model_weights)
    df_pred_proba = predict_proba(np.array(df_x_train_std),
                                  df_y_train,
                                  weights,
                                  df_y_train.unique().tolist())
    print(df_pred_proba.head())
    print("prediction are accurate at", 100 * np.mean(df_pred_proba['Accurate pred.']), "%.") 
    predict = df_pred_proba['Accurate pred.']
    df = pd.read_csv(f'./datasets/dataset_train.csv')
    df['Predicted outcome'] = df_pred_proba['Predicted outcome']
    df['Accurate pred.'] = predict
    print(df.head(10))
    df.drop(df.columns[0], inplace=True, axis = 1)
    df.to_csv(f'{model_dir}prediction_for_trainset1600.csv')
    df_inexact = df[df['Accurate pred.'] == 0]
    # sns.scatterplot(data=df_inexact, x="", y="Flying", hue="Best Hand", legend='auto')
    sns.scatterplot(data=df_inexact, x="Hogwarts House",y="Flying", hue="Best Hand", legend='auto')
    plt.show()
    # df_inexact.drop(df.columns[6:18], inplace=True, axis = 1)
    # df_inexact.sort_values(['First Name', 'Best Hand'], inplace=True)
    # df_inexact.to_csv(f'{model_dir}incorrect_prediction_for_trainset1600.csv')

    #sns.histplot(data=df, x="Accurate pred.'", color="skyblue", kde=True, hue="Hogwarts House")
    #sns.histplot(data=df, x="Herbology", color="skyblue", kde=True, hue="Accurate pred.")
         
if __name__ == "__main__":
    """ train model from file """
    main()
