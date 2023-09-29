import unittest
import numpy as np
import pandas as pd
from ..DescriberClass import Describe

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
        description = Describe(self.df)
        res = description.agg_describe(False)
        self.assertEqual(self.df.describe(), res)
        pass

if __name__ == '__main__':
    unittest.main()