import numpy as np
import pandas as pd
from utils import logreg_tools as logreg

"""
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
    """LogTrain class
    for multinomial logistic regression model training
    - one-vs-all strategy
    - gradient descent algorithm to minimize the error

    .train() method to start training the model
    """
    def __init__(self, df_x_train: pd.DataFrame,
                 df_class: pd.DataFrame):
        """ Parameters : unstandardized data to train without NaN, output """
        df_std = df_x_train.agg(lambda feat: logreg.standardize(feat))
        x_train_std = np.array(df_std)
        ones = np.ones((len(x_train_std), 1), dtype=float)
        self.x_train = np.concatenate((ones, x_train_std), axis=1)
        self.features = df_std.columns.tolist()
        self.df_class = df_class
        self.houses = df_class.unique().tolist()
        self._losses = pd.DataFrame(columns=self.houses)
        self._losses.fillna(0)
        w_indexes = df_x_train.columns.insert(0, ['Intercept'])
        self.df_weights = pd.DataFrame(columns=self.houses,
                                       index=w_indexes).fillna(0)

    @staticmethod
    def _loss_function(y_actual: np.ndarray,
                       h_pred: np.ndarray) -> float:
        """ returns : mean of losses.
        y_actual : target class. 1 in class, 0 not in class
        h_pred = sigmoid(x.weights)
        loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        """
        m = len(h_pred)
        a = - y_actual * np.log(h_pred)
        b = (1 - y_actual) * np.log(1 - h_pred)
        return (((a - b) / m)).mean()

    @staticmethod
    def _update_weight_loss(weights, learning_rate, grad_desc):
        return weights - learning_rate * grad_desc

    def _train_one_vs_all(self, house):
        """
        loss_iter = LogRegTrain._loss_function(y_actual, h_pred)
        gradient = np.dot(self.x_train.T, (h_pred - y_actual))
        """
        self.loss = []
        y_actual = np.where(self.df_class == house, 1, 0)
        weights = np.ones(len(self.features) + 1).T
        for iter in range(self.epochs):
            z_output = np.dot(self.x_train, weights)
            h_pred = logreg.sigmoid(z_output)
            tmp = np.dot(self.x_train.T, (h_pred - y_actual))
            grad_desc = tmp / y_actual.shape[0]
            weights = LogRegTrain._update_weight_loss(weights,
                                                      self.learning_rate,
                                                      grad_desc)
            self.loss.append(self._loss_function(y_actual, h_pred))
        return weights

    def train(self, learning_rate=0.1, epochs=1000):
        """

        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        for house in self.houses:
            weights = self._train_one_vs_all(house)
            self._losses[house] = self.loss
            self.df_weights[house] = weights
        print(f'Learning rate = {self.learning_rate}')
        print(f'Iterations = {self.epochs}')
        return self.df_weights

    def get_losses(self) -> pd.DataFrame:
        return self._losses

    def get_predict_proba(self) -> pd.DataFrame:
        """
        input is the trained dataset
        returns : a dataframe containing the probability for each outcome
        and the final predicted outcome
        """
        z = np.dot(self.x_train, self.df_weights)
        h = logreg.sigmoid(z)
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
