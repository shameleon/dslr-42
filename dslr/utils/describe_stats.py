#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
describe_stats.py functions are used by describe.py
 in dslr module, to avoid using numpy and pandas functions.
The only np and pd that are used :
    np.nan, np.sort(), np.array(), .to_list()

Usage:
    from utils import describe_stats as dum

    series_std = dum.std(my_series)

Test: functions for testing:
    __put_format_line() __put_format_line4()
    __test_utils_math() __test_utils_math_std():
"""
__authors__ = ['jmouaike, ebremond']

from math import floor, ceil, sqrt
from functools import reduce

import numpy as np
import pandas as pd


def is_nan(num) -> bool:
    return num != num


def is_not_nan(num) -> bool:
    return num == num


def drop_nan(x: pd.Series) -> pd.Series:
    return x[is_not_nan(x)]


def count_nan(x: pd.Series) -> int:
    return len(x[is_nan(x)])


def count(x: pd.Series) -> int:
    return len(drop_nan(x))


def sorted(x: pd.Series) -> np.ndarray:
    return np.sort(np.array(drop_nan(x)))


def add(a: float, b: float) -> float:
    return a + b


def min(x: pd.Series) -> float:
    if drop_nan(x).empty:
        return np.nan
    return sorted(x)[0]


def max(x: pd.Series) -> float:
    if drop_nan(x).empty:
        return np.nan
    return sorted(x)[-1]


def percentile(x: pd.Series, q: float) -> float:
    """
    Generic function for percentile calculation
    Parameters :
        x pd.Series is not neccessarily sorted
        q is the percentile in [0, 1] interval,
            q = 0 for minimum.
            q = 0.25 for first quartile,
            q = 0.5 for median,
            q = 0.75 for third quartile,
            q = 1 for maximum.

    variables:
        arr : sorted numpy array without NaNs
        g : is a virtual float index between two neighbors
        of indexes f and c.
    Return :
        float with a value that is a weighted average
        between the two neighbor values
    """
    if drop_nan(x).empty:
        return np.nan
    arr = sorted(x)
    g = (len(arr) - 1) * q
    f = floor(g)
    c = ceil(g)
    if f == c:
        return arr[int(g)]
    else:
        lower_neighbor = arr[int(f)] * (c - g)
        higher_neighbor = arr[int(c)] * (g - f)
        return lower_neighbor + higher_neighbor


def quantile_25(x: pd.Series) -> float:
    return percentile(x, 0.25)


def quantile_50(x: pd.Series) -> float:
    return percentile(x, 0.5)


def quantile_75(x: pd.Series) -> float:
    return percentile(x, 0.75)


def median(x: pd.Series) -> float:
    s = sorted(x)
    pos = len(s) // 2
    if len(s) % 2 == 0:
        return (s[pos - 1] + s[pos]) / 2
    else:
        return s[pos]


def sum(x: pd.Series) -> float:
    if drop_nan(x).empty:
        return 0.0
    list = drop_nan(x).to_list()
    x_sum = reduce(add, list)
    return x_sum


def mean(x: pd.Series):
    if drop_nan(x).empty:
        return np.nan
    return sum(x) / len(drop_nan(x))


def kth_moment(x: pd.Series, k: int, corrected=True) -> float:
    r"""k-th standardized moment of the population
    k should be in [1, 2, 3, 4] values
    :math:`m_k = \frac{1}{N-1}\sum_{i=1}^n (𝑥_i-\overline{𝑥})^k`
    """
    if count(x) <= corrected or k not in [1, 2, 3, 4]:
        return np.nan
    x_list = drop_nan(x).to_list()
    x_mean = mean(x)
    kth_deviation = 0
    for i in range(len(x_list)):
        kth_deviation += (x_list[i] - x_mean)**k
    return kth_deviation / (len(x_list) - corrected)


def std(x: pd.Series, corrected=True) -> float:
    r"""Standard deviation
    IMPORTANT : numpy std() is different than pandas std() !!!
    Pandas : DataFrame.describe() calls Series.std(),
        which is a corrected Standard deviation (Bessel's correction)
        and returns unbiased standard deviation over requested axis.
        Normalized by N-1 by default.
        Series.std(ddof=0) to obtain to normalize by N instead of N-1.

    Parameters:
        corrected=True : set by default,
            Return : a corrected std (normalized to N - 1) aswith pd.std()
        corrected=False: for an uncorrected std as np.std()
            Return : std (normalized to N) as np.std() or pd.std(ddof=1)

    standard_deviation:
    :math:`\sigma^2 = \frac{1}{N-1}\sum_{i=1}^n (𝑥_i-\overline{𝑥})^2`
    """
    if count(x) <= corrected:
        return np.nan
    return sqrt(kth_moment(x, 2, corrected))


def skewness(x: pd.Series) -> float:
    """sample skewness estimates symetry of a dataset
    a value of 0: symetric data, normally distributed
    negative value: skewed to the left
    positive value: skewed to the right

    french: coefficient d'asymétrie

    The third central moment calculation is applied

    Skewness:
    (sum of the Deviation Cube)/(N-1) * Std deviation’s Cube.
    """
    std_x = std(x)
    if count(x) <= 1 or std_x == 0 or is_nan(std_x):
        return np.nan
    return kth_moment(x, 3) / (std_x ** (3))


def kurtosis(x: pd.Series) -> float:
    """sample kurtosis is to estimate thickness of tails
    containing outliers of a distibution.
    value 3: normal distribution
    negative value < 3: less outliers
    positive value > 3 : more outlier
    equivalent to pd.kurt(), not pd.kurtosis()
    """
    sigma2 = kth_moment(x, 2)
    if sigma2 == 0 or is_nan(sigma2):
        return np.nan
    m4 = kth_moment(x, 4)
    sigma2 = kth_moment(x, 2)
    return m4 / sigma2**2


def __put_format_line(label: str, n1: float, n2: float) -> None:
    """ prints a width formatted line to stdout
    parameters : a label and two float numbers """
    line = "{:10}{: >20}{: >20}".format(label, n1, n2)
    print(line)


def __put_format_line4(label: str, n1, n2, n3, n4) -> None:
    """ prints a width formatted line to stdout
    parameters : a label and two float numbers """
    line = "{:10}{: >20}{: >20}{: >20}{: >20}".format(label, n1, n2, n3, n4)
    print(line)


def __test_utils_math(s: pd.Series):
    print("Sorted pd.Series", sorted(s))
    __put_format_line("", "np or pd", "dum")
    print("_" * 50)
    __put_format_line("count", s.count(), count(s))
    __put_format_line("min", s.min(), min(s))
    __put_format_line("25%", s.quantile(0.25), percentile(s, 0.25))
    __put_format_line("50%", s.quantile(0.5), percentile(s, 0.5))
    __put_format_line("75%", s.quantile(0.75), percentile(s, 0.75))
    __put_format_line("max", s.max(), max(s))
    __put_format_line("sum", np.sum(s), sum(s))
    __put_format_line("mean", np.mean(s), mean(s))
    __put_format_line("std", s.std(), std(s))
    print(f'{" bonus ":#^50}')
    __put_format_line("NaNs", len(s[s.isna()]), count_nan(s))
    __put_format_line("skewness", "%.6f" % s.skew(), "%.6f" % skewness(s))
    __put_format_line("kurtosis", "%.6f" % s.kurtosis(),
                      "%.6f" % kurtosis(s))
    print("_" * 50)


def __test_utils_math_std(df: pd.DataFrame):
    df2 = pd.DataFrame({'2+2': [1, np.nan, np.nan, 42],
                        '3+1': [50, 17, 42, np.nan]})
    df3 = pd.DataFrame({'2+0': [-1, 42],
                        '1+1': [np.nan, -42],
                        '0+2': [np.nan, np.nan]})
    df4 = pd.DataFrame({'1+0': [42],
                        '0+1': [np.nan]})
    df5 = pd.DataFrame({'0+0': []})
    print("_" * 75)
    __put_format_line4("std's",
                       "np.std()", "dum.std(False)",
                       "pd.std()", "dum.std(True)")
    series_to_test = [df['A'], df['B'], df['C'],
                      df2['2+2'], df2['3+1'],
                      df3['2+0'], df3['1+1'], df3['0+2'],
                      df4['1+0'], df4['0+1'], df5['0+0']]
    for s in series_to_test:
        __put_format_line4(s.name,
                           np.std(s), std(s, False),
                           s.std(), std(s, True))
    print("_" * 75)
    for s in series_to_test:
        __put_format_line4(s.name, "", "max :", s.max(), max(s))


if __name__ == "__main__":
    """Test: python describe_stats.py"""
    df = pd.DataFrame({
        'A': [42, np.nan, 3, np.nan, 5, 10, 18, 6, -2, 0],
        'B': [np.nan, 10, 9, 9, 9, 8, 9, 9, 9, np.nan],
        'C': [np.nan, np.nan, np.nan, np.nan, np.nan,
              np.nan, np.nan, np.nan, np.nan, np.nan]
        })
    print('TEST: utils/dslr_stats.py')
    __test_utils_math(df['B'])
    __test_utils_math(df['C'])
    __test_utils_math_std(df)
