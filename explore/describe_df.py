import argparse
import pandas as pd

""" program 
    """

def main():
    """ """
    df = pd.read_csv(f'./explore/cars.csv')
    print(df.head())
    print(df.info()) 
    print(df.describe())
    print('*' * 30)
    print([key for key in df.keys()])
    print('*' * 30)
    prices = df['price']
    print(prices.describe())


if __name__ == "__main__":
    main()

"""
https://github.com/alkashef/describe2
Definition and Usage

The describe() method returns description of the data in the DataFrame.

If the DataFrame contains numerical data, the description contains these information for each column:

count - The number of not-empty values.
mean - The average (mean) value.
std - The standard deviation.
min - the minimum value.
25% - The 25% percentile*.
50% - The 50% percentile*.
75% - The 75% percentile*.
max - the maximum value.

*Percentile meaning: how many of the values are less than the given percentile. Read more about percentiles in our Machine Learning Percentile chapter.
"""