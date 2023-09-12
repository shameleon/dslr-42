import numpy as np
import os
import pandas as pd
from MultinomialTrainClass import LogRegTrain

"""training.py:

Programm to train a linear regression model
based on technique of gradient descent to minimize the error

a car price to mileage datasetdata.csv is required

"""

__author__ = "jmouaike"

def drop_useless(df: pd.DataFrame) -> pd.DataFrame:
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
    return df.dropna()


def main() -> None:
    """ df : read from file
        df_train : kept only target variable columns 
        and meaningful data as unstandardized data 
        target : target variable
        """
    model_dir = './logistic_reg_model/'
    dest_file = 'gradient_descent_weights.csv'
    df = pd.read_csv(f'./datasets/dataset_train.csv')
    df_train = drop_useless(df)
    # 5 rows x 12 columns
    target = df_train.columns[1]
    # features = df_train.columns[2:]
    # df_classes = df_train[target]
    # houses = df_classes.unique().tolist()
    # extract features columns
    df_x_train = df_train[df_train.columns[2:]]
    # mean_x = df_x_train.mean(axis=0)
    # std_x = df_x_train.std(axis=0)
    df_normalizing = pd.DataFrame({'mean_x': df_x_train.mean(axis=0),
                                 'std_x': df_x_train.std(axis=0)}).T
    df_normalizing.to_csv(f'{model_dir}normalization.csv')
    #apply(lambda x: np_mean(x))
    # 5 rows x 10 columns - only features
    # extract output column
    df_class = df_train[target]
    # (1333, )
    logreg_model = LogRegTrain(df_x_train, df_class)
    df_weights = logreg_model.train(0.1, 1000)
    # print(df_weights)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    #no headers : np.savetxt(f'{model_dir + dest_file}', df_weights, delimiter=",")
    df_weights.to_csv(f'{model_dir + dest_file}')
    # predicted proba on trained set (normalized and dropna-ed)
    df_pred_proba = logreg_model.get_predict_proba()
    df_pred_proba.to_csv(f'{model_dir}prediction_for_trainset1333.csv')
    #print(df_pred_proba.head(20))
    print("prediction are accurate at", 100 * np.mean(df_pred_proba['Accurate pred.']), "%.") 
    
    return None

if __name__ == "__main__":
    """
    train model from file ../datasets/dataset_train.csv 
    """
    main()


"""
                                Accuracy
logreg_model.train(0.1, 1000)   98.2745 %
logreg_model.train(0.1, 5000)   98.2745 %
logreg_model.train(0.01, 1000)   98.2745 %
logreg_model.train(0.01, 5000)   98.2745 %

"""