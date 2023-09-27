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
    x_mean = mean(x)
    sq_sum = 0
    for i in range(len(x_list)):
        sq_sum += (x_list[i]- x_mean)**2
        sqrt(sq_sum / len(x_list))
    return sqrt(sq_sum / len(x_list))

def test_utils_math(s: pd.Series):
    print(sorted(s))
    print("__________________________________")
    print("       ", "np or pd", "|", "utils.math")
    print("count    ", s.count(), "   |  ", count(s))
    print("min     ", s.min(), " | ", min(s))
    print("25%      ", s.quantile(0.25), " |  ", percentile(s, 0.25))
    print("50%      ", s.quantile(0.5), " |  ", percentile(s, 0.5))
    print("75%      ", s.quantile(0.75), "|  ", percentile(s, 0.75))
    print("max      ", s.max(), "|  ", max(s))
    print("NaNs     ", len(s[s.isna()]), "   |  ", count_nan(s))
    print("__________________________________")
    print("sum     ", np.sum(s), "   |  ", sum(s))
    print("mean     ", np.mean(s), "   |  ", mean(s))
    print("std     ", np.std(s), "   |  ", std(s))


# if __name__ == "__main__":
#     # create a DataFrame with missing values
#     df = pd.DataFrame({
#     'A': [42, np.nan, 3, np.nan, 5, 10, 18, 6, -2, 0],
#     'B': [np.nan, 4, 1, 4, 5, 8, 0, 12, 20, np.nan],
#     'C': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     })
#     # test_utils_math(df['A'])
#     test_utils_math(df['B'])
#     # test_utils_math(df['C'])

#     # for q in np.arange(0, 1.25, 0.25):
#     #     print(q, percentile(df['C'], q))


"""
NaN :
https://stackoverflow.com/questions/944700/how-to-check-for-nan-values
"""