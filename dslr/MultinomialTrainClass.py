import numpy as np
import pandas as pd
#import stat_utlis as su

"""training.py:

class for multinomial logistic regression model training
        - one-vs-all strategy
        - gradient descent algorithm to minimize the error

Multinomial Logistic regression : Where the target variable has 
three or more possible classes. 

One-Vs-All Classification is a method of multi-class classification.
It can be broken down by splitting up the multi-class classification
problem into multiple binary classifier models. 
For k class labels present in the dataset, k binary classifiers
are needed in One-vs-All multi-class classification.
"""

__author__ = "jmouaike"

class LogRegTrain:
    """
    """
    def __init__(self, df_x_train, df_class):
        """ Parameters : unstandardized data to train without NaN, output """
        df_std = df_x_train.agg(lambda feature: LogRegTrain.standardize(feature)) 
        x_train_std = np.array(df_std)
        ones = np.ones((len(x_train_std), 1), dtype=float)
        self.x_train = np.concatenate((ones, x_train_std), axis=1)  
        self.features = df_std.columns.tolist()
        self.df_class = df_class
        self.houses = df_class.unique().tolist()
        # self.df_losses = pd.DataFrame(columns=self.houses)
        # self.df_losses.fillna(0)
        w_indexes = df_x_train.columns.insert(0, ['Intercept'])
        self.df_weights = pd.DataFrame(columns=self.houses, index=w_indexes).fillna(0)

    @staticmethod
    def standardize(arr:np.ndarray):
        mean = np.mean(arr)
        std = np.std(arr)
        return (arr - mean) / std
    
    @staticmethod
    def sigmoid(arr:np.ndarray):
        return 1 / (1 + np.exp(-arr))
    
    @staticmethod
    def sigmoid(n):
        return 1 / (1 + np.exp(-n))
    
    @staticmethod
    def loss_function(y_actual, h_pred):
        """ y_actual : target class. 1 in class, 0 not in class
        h_pred = signoid(x.weights)
        loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        """
        m = len(h_pred)
        a = -y_actual * np.log(h_pred)
        b = (1 - y_actual) * np.log(1 - h_pred)
        return (a - b) / m
    
    # @staticmethod
    # def gradient_descent(x_train, h_pred, y_actual):
    #     return np.dot(x_train.T, (h_pred - y_actual)) / y_actual.shape[0]

    @staticmethod
    def update_weight_loss(weights, learning_rate, grad_desc):
        return weights - learning_rate * grad_desc
    
    def train_one_vs_all(self, house):
        y_actual = np.where(self.df_class == house, 1, 0)
        #loss = []
        weights = np.ones(len(self.features) + 1).T
        #weights = np.zeros(len(self.features) + 1).T
        for iter in range(self.epochs):
            z_output = np.dot(self.x_train, weights)
            h_pred = LogRegTrain.sigmoid(z_output) 
            loss_iter = LogRegTrain.loss_function(y_actual, h_pred)
            #loss.append(loss_iter)
            gradient = np.dot(self.x_train.T, (h_pred - y_actual))
            # grad_desc = LogRegTrain.gradient_descent(self.x_train, h_pred, gradient)
            tmp = np.dot(self.x_train.T, (h_pred - y_actual))
            grad_desc = tmp / y_actual.shape[0]
            weights = LogRegTrain.update_weight_loss(weights, self.learning_rate, grad_desc)
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
        # odds = np.log(z)
        h = LogRegTrain.sigmoid(z)
        df_pred_proba = pd.DataFrame(h, columns=self.houses)
        df_pred_proba['Predicted outcome'] = df_pred_proba.idxmax(axis=1)
        df_pred_proba['Real outcome'] = self.df_class.tolist()
        df_pred_proba['Accurate pred.'] = np.where(df_pred_proba['Predicted outcome'] 
                                         == df_pred_proba['Real outcome'], 1, 0)
        return df_pred_proba