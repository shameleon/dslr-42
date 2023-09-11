import numpy as np
import pandas as pd

"""
"""

__author__ = "jmouaike"


def standardize(arr:np.ndarray):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std


def sigmoid(arr:np.ndarray):
    return 1 / (1 + np.exp(-arr))


def loss_function(y_actual, h_pred):
    """ y_actual : target class. 1 in class, 0 not in class
    h_pred = signoid(x.weights)
    loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    """
    m = len(h_pred)
    a = -y_actual * np.log(h_pred)
    b = (1 - y_actual) * np.log(1 - h_pred)
    return (a - b) / m


def gradient_descent(x_train, h_pred, y_actual):
    return np.dot(x_train.T, (h_pred - y_actual)) / y_actual.shape[0]


def update_weight_loss(weights, learning_rate, grad_desc):
    return weights - learning_rate * grad_desc
    