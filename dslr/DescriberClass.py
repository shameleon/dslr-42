import numpy as np
import pandas as pd
from utils import dslr_stats as dum

"""DescriberClass.py:
    - class Describe
    - tests for the class

    Class Describe reproduces the behavior of pandas library
    describe() function for the mandatory part of 42's subject.
    
    Bonus:
    More statistics lines are provided if bonus=True 
    for self.agg_describe() method.
"""

__author__ = "jmouaike"

class   Describe:
    """ Reproduces the behavior of pandas pd.describe() function
    Parameter : a pandas dataframe
    Output : displays information for all numerical features of the dataframe"""
    def __init__(self, df=pd.DataFrame):
        self.df_num = df[df.select_dtypes(include=np.number).columns]
        self.funcs = [dum.count, dum.mean, dum.std, dum.min, dum.quantile_25,
                 dum.quantile_50, dum.quantile_75, dum.max]
        self.bonus_funcs = [dum.count_nan]

    def agg_describe(self, bonus=False):
        """ using custom functions and counting NaN
        [9 rows x 13 columns]"""
        new_idx_names = {'quantile_25': '25%',
                         'quantile_50': '50%',
                         'quantile_75': '75%'}
        if self.df_num.empty:
            return None
        if bonus:
            self.funcs += self.bonus_funcs
        self.description = self.df_num.agg(self.funcs)
        self.description.rename(index=new_idx_names, inplace=True)
        print(self.description.map(lambda x: f"{x:0.6f}"))
        return self.description


def test_describe_class():
    """  """
    df = pd.DataFrame({'categorical': pd.Categorical(['d','e','f', 'g']),
                       'feature1': [1, 22, 21, 42],
                       'feature2': [50, 17, 42, 23]})
    print("TEST : Describe class")
    pandas_stats = df.describe()
    print("pandas describe() :")
    print(df.describe())
    description = Describe(df)
    print("dslr project describe() :")
    stats = description.agg_describe(False)
    print("Results :")
    result = (stats == pandas_stats)
    print(result)


if __name__ == "__main__":
    test_describe_class()


"""
pd.describe() describes numeric columns including Index

count
mean
std
min
25%
50%
75%
max

empty_dataframe.describe()

count   0
unique  0
top     NaN
freq    NaN

https://inside-machinelearning.com/skewness-et-kurtosis/
    Skewness
    Kurtosis
    No. of 3 Sigma Outliers

"""