import numpy as np
import pandas as pd
import utils.math as dum

"""DescriberClass.py:
    - class Describe
    - tests for the class

    Class Describe reproduces the behavior of pandas describe()
    function for the mandatory part of 42's subject.
    
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
        self.df = df
        self.df_num = self.df[df.select_dtypes(include=np.number).columns[1:]]
        self.funcs = [dum.count, dum.mean, dum.std, dum.min, dum.quantile_25,
                 dum.quantile_50, dum.quantile_75, dum.max]
        self.bonus_funcs = [dum.count_nan]

    def agg_describe(self, bonus=False):
        """ using custom functions and counting NaN
        [9 rows x 13 columns]"""
        if self.df.empty:
            return None
        if bonus:
            self.funcs += self.bonus_funcs
        description = self.df_num.agg(self.funcs)
        print(description.applymap(lambda x: f"{x:0.2f}"))


def test_describe():
    """ https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html# """
    print("\npd.Series([1, 2, 3]  ===> describe()")
    s = pd.Series([1, 2, 3])
    print(s.describe())
    df = pd.DataFrame({'categorical': pd.Categorical(['d','e','f']),
                       'numeric': [1, 2, 3],
                       'object': ['a', 'b', 'c']})
    print("\npd.DataFrame ===> describe()")
    df.describe(include='all')
    print(df.numeric.describe())


if __name__ == "__main__":
    test_describe()


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