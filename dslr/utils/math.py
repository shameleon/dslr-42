from math import floor, ceil, sqrt
from functools import reduce 
import numpy as np
import pandas as pd

"""
utils.math.py are functions primarily intended
to be used by describe.py:

the purpose is to avoid using numpy and pandas

functions used : 
np.sort()
np.array()
.to_list()
"""

__author__ = "jmouaike"


def is_nan(num):
    return num != num


def is_not_nan(num):
    return num == num


def drop_nan(x: pd.Series):
    return x[is_not_nan(x)]


def count_nan(x):
    return len(x[is_nan(x)])


def count(x):
    return len(drop_nan(x))


def sorted(x: pd.Series) -> np.ndarray:
    return np.sort(np.array(drop_nan(x)))


def min(x: pd.Series):
    return sorted(x)[0]


def max(x: pd.Series):
    return sorted(x)[-1]


def percentile(x: pd.Series, q:float):
    """
    Parameters : 
    q is the percentile in [0, 1] interval,
    q = 0 for minimum.
    q = 0.25 for first quartile, 
    q = 0.5 for median, 
    q = 0.75 for third quartile,
    q = 1 for maximum.
    
    Return :
        float value
    arr : sorted numpy array without NaNs
    """
    arr = sorted(x)
    g = (len(arr) - 1) * q
    f = floor(g)
    c = ceil(g)
    if f == c:
        return arr[int(g)]
    else:
        lower_neighbor = arr[int(f)] * (c - g)
        higher_neighbor = arr[int(c)] * (g -f)
        return lower_neighbor + higher_neighbor


def quantile_25(x: pd.Series):
    return percentile(x, 0.25)


def quantile_50(x: pd.Series):
    return percentile(x, 0.5)


def quantile_75(x: pd.Series):
    return percentile(x, 0.75)


def median(x: pd.Series):
    s = sorted(x)
    pos = len(s) // 2
    if len(s) % 2 == 0:
        return (s[pos - 1] + s[pos]) / 2
    else:
        return s[pos]


def sum(x: pd.Series):
    list = drop_nan(x).to_list()
    f = lambda a, b: a + b
    x_sum = reduce(f, list)
    return x_sum


def mean(x: pd.Series):
    return sum(x) / len(drop_nan(x))


def std(x: pd.Series):
    """ Standard deviation"""
    x_list = drop_nan(x).to_list()
    if len(x_list) == 0:
        return np.nan
    x_mean = mean(x)
    sq_sum = 0
    for i in range(len(x_list)):
        sq_sum += (x_list[i] - x_mean)**2
    return sqrt(sq_sum / len(x_list))


def put_wline(label: str, n1, n2):
    line = "{:10}{: >20}{: >20}".format(label, n1, n2)
    print(line)


def test_utils_math(s: pd.Series):
    print("sorted", sorted(s))
    put_wline("", "np or pd", "utils.math")
    print("_" * 50)
    put_wline("count", s.count(), count(s))
    put_wline("min", s.min(), min(s))
    put_wline("25%", s.quantile(0.25), percentile(s, 0.25))
    put_wline("50%", s.quantile(0.5), percentile(s, 0.5))
    put_wline("75%", s.quantile(0.75), percentile(s, 0.75))
    put_wline("max", s.max(), max(s))
    put_wline("NaNs", len(s[s.isna()]), count_nan(s))
    put_wline("sum", np.sum(s), sum(s))
    put_wline("mean", np.mean(s), mean(s))
    put_wline("std", np.std(s), std(s))
    print("_" * 50)
    df2 = pd.DataFrame({'feature1': [1, np.nan, np.nan, 42],
                    'feature2': [50, 17, 42, np.nan]})
    put_wline("std_feature1", np.std(df2['feature1']), std(df2['feature1']))
    put_wline("std_feature1", np.std(df2['feature2']), std(df2['feature2']))

if __name__ == "__main__":
    # create a DataFrame with missing values
    df = pd.DataFrame({
    'A': [42, np.nan, 3, np.nan, 5, 10, 18, 6, -2, 0],
    'B': [np.nan, 4, 1, 4, 5, 8, 0, 12, 20, np.nan],
    'C': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    })
    test_utils_math(df['B'])

"""
NaN :
https://stackoverflow.com/questions/944700/how-to-check-for-nan-values
"""