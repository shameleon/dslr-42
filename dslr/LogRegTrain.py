import numpy as np
import pandas as pd

"""training.py:

class for multinomial logistic regression model training
        - one-vs-all strategy
        - gradient descent algorithm.
"""

__author__ = "jmouaike"

class LogRegTrain:
    def __init__(self, df_x_train, df_class):
        df_std = df_x_train.agg(lambda feature: LogRegTrain.standardize(feature)) 
        x_train_std = np.array(df_std)
        # add a column of ones at begin
        ones = np.ones((len(x_train_std), 1), dtype=float)
        self.x_train = np.concatenate((ones, x_train_std), axis=1)  
        self.features = df_std.columns.tolist()
        self.df_class = df_class
        #self.houses = ['Ravenclaw', 'Slytherin', 'Griffondor', 'Hufflepuff']
        self.houses = df_class.unique().tolist()
        # self.df_losses = pd.DataFrame(columns=self.houses)
        # self.df_losses.fillna(0)
        w_indexes = df_x_train.columns[2:].insert(0, ['Bias'])
        self.df_weights = pd.DataFrame(columns=self.houses, index=w_indexes).fillna(0)

    