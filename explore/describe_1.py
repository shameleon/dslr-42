import numpy as np
import pandas as pd
import sys

"""describe.py:

Program called describe.[extension].
This program will take a dataset as a parameter

it displays information for all numerical features 
"""

__author__ = "jmouaike"

class   DescribeData:
    """ """
    def __init__(self):
        self.file = sys.argv[1]
        self.df = pd.DataFrame({'A' : []})
        self.load_data()
        self.df_num = self.df[self.df.select_dtypes(include=np.number).columns[1:]]

    @staticmethod
    def quantile_25(x):
        return x.quantile(0.25)
    
    @staticmethod
    def quantile_50(x):
        return x.quantile(0.5)
    
    @staticmethod
    def quantile_75(x):
        return x.quantile(0.25)
    
    @staticmethod
    def nan_count(x):
        return x.isna().sum()

    def agg_describe(self):
        """ using custom functions and coutning NaN
        [9 rows x 13 columns]"""
        if self.df.empty:
            return None
        describer2 = self.df_num.agg(['count', self.nan_count, 'mean', 'std', 'min',
                self.quantile_25, self.quantile_50, self.quantile_75, 'max'])
        print(describer2.applymap(lambda x: f"{x:0.2f}"))

    def test_standard_describe(self):
        """ https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html# 
        [8 rows x 13 columns]"""
        if self.df.empty:
            return None
        describer = self.df_num.describe()
        #.apply(lambda x: "{x:.2f}".format)
        print(describer.applymap(lambda x: f"{x:0.2f}"))


    def load_data(self):
        try:
            self.df = pd.read_csv(f'{sys.argv[1]}')
            print(self.df.head())
        except (AttributeError):
            print(f'Error: attribute error')
        except (IndexError):
            print(f'Usage: \'{sys.argv[1]}\' <filename.csv>')
        except (FileNotFoundError, IsADirectoryError, pd.errors.ParserError):
            print(f'Error: \'{sys.argv[1]}\' is not a valid .csv')


if __name__ == "__main__":
    if len(sys.argv[1:]) == 1:
        dataset_describer = DescribeData()
        dataset_describer.test_standard_describe()
        dataset_describer.agg_describe()
        # try:
        #     dataset_describer = DescribeData()
        #     dataset_describer.test_standard_describe()
        # except (AttributeError):
        #     print(f'Error: Unsuitable dataset for describe.py')
