import numpy as np
import pandas as pd
import config
from utils import logreg_tools as logreg

"""
"""

__author__ = "jmouaike"


class PredictFromLogRegModel:
    def __init__(self) -> None:
        self.target = config.target


    def predict_proba(self, df_test_std: pd.DataFrame, model_weights: pd.DataFrame) -> pd.DataFrame:
        x_test = np.array(df_test_std)
        ones = np.ones((len(x_test), 1), dtype=float)
        x_test = np.concatenate((ones, x_test), axis=1)
        weights = np.array(model_weights.drop(columns=features_label))
        classifiers = model_weights.columns[1:].to_list()
        z = np.dot(x_test, weights)
        h = sigmoid(z)
        y_pred_proba = pd.DataFrame(h, columns=classifiers)
        return y_pred_proba