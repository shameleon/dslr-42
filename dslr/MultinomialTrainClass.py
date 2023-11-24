import numpy as np
import pandas as pd
from utils import describe_stats as dum

"""
class for multinomial logistic regression model training
    - one-vs-all strategy
    - gradient descent algorithm to minimize the error

Multinomial Logistic regression : Where the target variable has
three or more possible classes.

One-Vs-All Classification : a method of multi-class classification,
where the multi-class classification problem is broken down into
multiple binary classifier models.
For k class labels present in the dataset, k binary classifiers
are needed in One-vs-All multi-class classification.
"""

__author__ = "jmouaike"


class LogRegTrain:
    """
    """
    def __init__(self, df_x_train, df_class):
        """ Parameters : unstandardized data to train without NaN, output """
        df_std = df_x_train.agg(lambda feat: LogRegTrain.standardize(feat))
        x_train_std = np.array(df_std)
        ones = np.ones((len(x_train_std), 1), dtype=float)
        self.x_train = np.concatenate((ones, x_train_std), axis=1)
        self.features = df_std.columns.tolist()
        self.df_class = df_class
        self.houses = df_class.unique().tolist()
        # self.df_losses = pd.DataFrame(columns=self.houses)
        # self.df_losses.fillna(0)
        w_indexes = df_x_train.columns.insert(0, ['Intercept'])
        self.df_weights = pd.DataFrame(columns=self.houses,
                                       index=w_indexes).fillna(0)

    @staticmethod
    def standardize(arr: np.ndarray) -> np.ndarray:
        """standardization with Z-score"""
        mean = dum.mean(arr)
        std = dum.std(arr)
        return (arr - mean) / std

    @staticmethod
    def sigmoid(arr: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-arr))

    @staticmethod
    def loss_function(y_actual: np.ndarray,
                      h_pred: np.ndarray) -> np.ndarray:
        """ y_actual : target class. 1 in class, 0 not in class
        h_pred = signoid(x.weights)
        loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        """
        m = len(h_pred)
        a = - y_actual * np.log(h_pred)
        b = (1 - y_actual) * np.log(1 - h_pred)
        return (a - b) / m

    @staticmethod
    def update_weight_loss(weights, learning_rate, grad_desc):
        return weights - learning_rate * grad_desc

    def train_one_vs_all(self, house):
        """
        loss_iter = LogRegTrain.loss_function(y_actual, h_pred)
        gradient = np.dot(self.x_train.T, (h_pred - y_actual))
        """
        y_actual = np.where(self.df_class == house, 1, 0)
        weights = np.ones(len(self.features) + 1).T
        for iter in range(self.epochs):
            z_output = np.dot(self.x_train, weights)
            h_pred = LogRegTrain.sigmoid(z_output)
            tmp = np.dot(self.x_train.T, (h_pred - y_actual))
            grad_desc = tmp / y_actual.shape[0]
            weights = LogRegTrain.update_weight_loss(weights,
                                                     self.learning_rate,
                                                     grad_desc)
        return weights


    def train(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        for house in self.houses:
            weights = self.train_one_vs_all(house)
            # self.df_losses[house] = loss
            self.df_weights[house] = weights
        print(self.learning_rate, "  iterations =", self.epochs)
        return self.df_weights


    def get_predict_proba(self) -> pd.DataFrame:
        """
        input is the trained dataset
        returns : a dataframe containing the probability for each outcome
        and the final predicted outcome
        """
        print(self.x_train.shape, self.df_weights.shape)
        z = np.dot(self.x_train, self.df_weights)
        h = LogRegTrain.sigmoid(z)
        df_pred_proba = pd.DataFrame(h, columns=self.houses)
        df_pred_proba['Predicted outcome'] = df_pred_proba.idxmax(axis=1)
        df_pred_proba['Real outcome'] = self.df_class.tolist()
        accurate_pred = np.where(df_pred_proba['Predicted outcome']
                                 == df_pred_proba['Real outcome'], 1, 0)
        df_pred_proba['Accurate pred.'] = accurate_pred
        return df_pred_proba


if __name__ == "__main__":
    """_summary_
    """
    pass

    