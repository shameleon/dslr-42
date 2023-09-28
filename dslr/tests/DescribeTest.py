import unittest
import numpy as np
import pandas as pd

class DescribeTesting(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        try:
            df = pd.read_csv("./datasets/dataset_train.csv")
        except (FileNotFoundError, IsADirectoryError) as e:
            print("File Error :", e)
        except pd.errors.EmptyDataError as e:
            print("File Content Error :", e)
        if df.empty:
            self.df = None
        else:
            self.df = df.columns[1:]

    def test_describe(self):
        self.df.describe()
        pass

if __name__ == '__main__':
    unittest.main()