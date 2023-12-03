#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dslr/testDescribe.py

Unit testing for dslr/DescriberClass Describe class

"""

import unittest
import pandas as pd

from DescriberClass import Describe


class DescribeTesting(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        try:
            self.df = pd.read_csv("./datasets/dataset_train.csv")
        except (FileNotFoundError, IsADirectoryError) as e:
            print("File Error :", e)
        except pd.errors.EmptyDataError as e:
            print("File Content Error :", e)
        if self.df.empty:
            self.df = None
        else:
            self.df.drop(self.df.columns[0], inplace=True, axis=1)

    def test_describe(self):
        """ test that compares Describe.agg_describe()
        to pandas describe.
        rtol : Relative tolerance.
        """
        rtol = 1e-12
        description = Describe(self.df)
        ours = description.agg_describe(False).to_numpy()
        theirs = self.df.describe().to_numpy()
        diff = ours - theirs
        print(f'\n{(diff > rtol).sum()} errors in diff at {rtol} precision')
        self.assertTrue((diff > rtol).sum() == 0)


if __name__ == '__main__':
    unittest.main()
