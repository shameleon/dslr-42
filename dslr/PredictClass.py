import numpy as np
import pandas as pd
import config
from utils import logreg_tools as logreg
from utils import print_out as po

"""
"""

__author__ = "jmouaike"


class PredictFromLogRegModel:
    """ """
    def __init__(self, df: pd.DataFrame, weights: pd.DataFrame) -> None:
        self.target = config.target_label
        self.df = df
        self.model_weights = weights
        features_label = weights.columns[0]
        self.model_features = weights[features_label].to_list()[1:]
        self.x_test = self._init_x_test(df)
        self.weights = np.array(weights.drop(columns=features_label))
        self.result = self._predict_outputs()

    def _init_x_test(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and transform data from dataframe
           and returns an array ready for prediction.
        - keeping only features that are in the model
        - standardization with the z-score method
        - replacing NaNs with 0 (= mean of the feature)
        - adding a column of ones for later dot product
        with intercept weights.
        """
        df_test = df[self.model_features]
        df_test_std = df_test.agg(lambda feat: logreg.standardize(feat))
        df_test_std.fillna(0, inplace=True)
        x_test = np.array(df_test_std)
        ones = np.ones((len(x_test), 1), dtype=float)
        return np.concatenate((ones, x_test), axis=1)

    def _predict_outputs(self) -> pd.DataFrame:
        classifiers = self.model_weights.columns[1:].to_list()
        z = np.dot(self.x_test, self.weights)
        h = logreg.sigmoid(z)
        y_pred = pd.DataFrame(h, columns=classifiers)
        y_pred['Best Probability'] = y_pred.max(axis=1)
        y_pred['Predicted Outcome'] = y_pred.idxmax(axis=1)
        return y_pred

    def compare_to_truth(self, truth: pd.DataFrame):
        po.as_check('Model predictions compared to real outcomes')
        self.result['Real Outcome'] = truth.to_list()
        is_same = np.where(self.result['Predicted Outcome']
                           == self.result['Real Outcome'], 1, 0)
        self.result['Accurate Prediction'] = is_same
        accuracy = self.get_accuracy()
        po.as_result(f'Accuracy for the tested dataset: {accuracy * 100}%\n')
        return None

    def inaccurate_prediction_details(self):
        inaccurate = self.result[self.result['Accurate Prediction'] == 0]
        print("Inaccurate prediction details :")
        print(inaccurate.head(10))
        print(self.df.iloc[inaccurate.index.to_list()].head(10))

    def get_accuracy(self) -> float:
        return self.result['Accurate Prediction'].value_counts(1)[1]


if __name__ == "__main__":
    """_summary_
    """
    pass
