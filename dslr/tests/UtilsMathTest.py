import numpy as np
import pandas as pd
import unittest
from ..utils import math as dum

""" Unit testing for utils/math.py """


class UtilsMathTesting(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.df = pd.DataFrame({
            'A': [42, np.nan, 3, np.nan, 5, 10, 18, 6, -2, 0],
            'B': [np.nan, 4, 1, 4, 5, 8, np.nan, 12, 20, np.nan],
            'C': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            })
        self.s = self.df['B']

    def test_count(self):
        s = self.s
        self.assertEqual(s.count(), dum.count(s))
    
    def test_min(self):
        s = self.s
        self.assertEqual(s.min(), dum.min(s))

    def test_quantile_25(self):
        s = self.s
        self.assertEqual(s.quantile(0.25), dum.percentile(s, 0.25))
                         
    def test_quantile_50(self):
        s = self.s
        self.assertEqual(s.quantile(0.5), dum.percentile(s, 0.5))

    def test_quantile_75(self):
        s = self.s
        self.assertEqual(s.quantile(0.75), dum.percentile(s, 0.75))
    
    def test_max(self):
        s = self.s
        self.assertEqual(s.max(), dum.max(s))

    def test_NaN_count(self):
        s = self.s
        self.assertEqual(len(s[s.isna()]), dum.count_nan(s))

    def test_sum(self):
        s = self.s
        self.assertEqual(np.sum(s), dum.sum(s))

    def test_mean(self):
        s = self.s
        self.assertEqual(np.mean(s), dum.mean(s))

    def std(self):
        s = self.s
        self.assertEqual(np.std(s), dum.std(s), "KO")


if __name__ == '__main__':
    """python3 -m unittest -v ./dslr/tests/UtilsMathTest.py
    not forgetting -v --verbose option for detailled output
    
    https://docs.python.org/fr/3/library/unittest.html
    """

    unittest.main()