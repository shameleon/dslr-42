import argparse
import os
import pandas as pd
import sys
import config
from PredictClass import PredictFromLogRegModel
from utils import print_out as po

"""predict_dataset.py"""

__author__ = "jmouaike"


def set_real_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    predict_dataset.py searches for the real output for the classifiers
    in the Dataframe.
    if not found, it looks for truth file defined in config.py
    in the same directory that the tested dataframe.
    config.py : 'target_label' entry defines truth file name.

    Parameter : Dataframe to test
    Returns : the real output,
            so that the index can be used to compared
            with predicted ouput
    """
    target = config.target_label
    if df[target].dropna().shape[0] > 0:
        df_real_class = df[target]
    else:
        dir = os.path.split(args.test_file_path)[0]
        truth_file = os.path.join(dir, config.test_truth)
        df_real_class = pd.read_csv(truth_file)[target]
        if df_real_class.shape[0] == 0:
            raise SystemExit(1)
    return df_real_class


def test_dataset():
    """ Reading files from locations given as arguments """
    try:
        df = pd.read_csv(args.test_file_path)
        df_weights = pd.read_csv(args.weights_file_path)
        test_model = PredictFromLogRegModel(df, df_weights)
        df_real_class = set_real_class(df)
        po.as_title(f'Predicting {config.target_label}')
        po.as_title2(f'Sample size : {df.shape[0]}')
        test_model.compare_to_truth(df_real_class)
    except (FileNotFoundError, IsADirectoryError) as e:
        po.as_error("File Error :", e)
        sys.exit(1)
    except pd.errors.EmptyDataError as e:
        po.as_error("File Content Error :", e)
        sys.exit(1)


if __name__ == "__main__":
    """ train model from file """
    parser = argparse.ArgumentParser(prog='logreg_precit.py',
                                     description='Testing dataset',
                                     epilog='Enter a valid csv file, please')
    parser.add_argument('test_file_path')
    parser.add_argument('weights_file_path')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-b', '--bonus', action='store_true')
    args = parser.parse_args()
    test_dataset()
    sys.exit(0)
