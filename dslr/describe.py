import pandas as pd

"""describe.py:

Program called describe.[extension].
This program will take a dataset as a parameter

it displays information for all numerical features 
"""

__author__ = "jmouaike"

class   Describe:
    """ """
    def __init__(self):
        pass

    def extension(self):
        pass


def test_describe():
    """ https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html# """
    s = pd.Series([1, 2, 3])
    s.describe()
    df = pd.DataFrame({'categorical': pd.Categorical(['d','e','f']),
                       'numeric': [1, 2, 3],
                       'object': ['a', 'b', 'c']})
    df.describe(include='all')
    df.numeric.describe() 


if __name__ == "__main__":
    test_describe()


"""
Numerical column stats:

    Type
    Density
    No. of Values
    No. of Unique Values
    No. of NaN
    No. of Zeros
    No. of +ve Values
    Min
    Max
    Mean
    Std
    Median
    Mode
    Skew
    Kurtosis
    No. of 3 Sigma Outliers

Object column stats:

    Type
    Density
    No. of Values
    No. of Unique Values No. of NaN
"""