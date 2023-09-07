import numpy as np
import pandas as pd

"""
pandas 10 min tutorial 
https://pandas.pydata.org/docs/user_guide/10min.html#min

    
    # Bin the Age column into 3 equal-sized bins
binning numerical columns
data['UnitGroup'] = pd.cut(data['Unit'], bins=3)

# bin the numerical column into 3 groups
qcut(), is a function that divides a set of values
into bins according to the sample quantiles. 
data['QcutBin'] = pd.qcut(data['Unit'], q=3)

df.sort_values(by="B")
"""

def subsets(df: pd.DataFrame):
    print('#' * 30, "subsets", '#' * 30)
    df = pd.read_csv(f'./explore/cars.csv')
    print("price over 10k :")
    print(df[df['price'] > 10000])
    print('=' * 30)
    #Setting by assigning with a NumPy array:
    df.loc[:, 'grade'] = np.array([2] * len(df))
    print("select specific mades :")
    print(df[df['made'].isin(['Toyota', 'Subaru'])])
    print("mean by made :")
    df1 = df.groupby('made')[['mileage', 'price']].mean()
    print(df1)
    ""

def apply_function(df: pd.DataFrame):
    print('#' * 30, "apply#1", '#' * 30)
    print(df.describe())
    print('=' * 30)
    df_numerics_only = df.select_dtypes(include=np.number)
    print(df_numerics_only)
    print('=' * 30)
    # DataFrame.agg() and DataFrame.transform() 
    # applies a user defined function that reduces or broadcasts its result respectively.
    df2 = df_numerics_only .agg(lambda x: np.mean(x))
    print(df2)


def apply_function2(df: pd.DataFrame):
    """ KO """
    print('#' * 30, "apply#2", '#' * 30)
    print(df.describe())
    print('=' * 30)
    df_numerics_only = df.select_dtypes(include= np.number)
    print(df_numerics_only)
    for key in df.keys():
        print(np.issubdtype(df[key].dtype, np.number)) # remove cols where NaN
        # tmp.notna()
        #print(key, tmp.dtype())
        #print(tmp.dtype() == 'int64')

def apply_function3(df: pd.DataFrame):
    print('#' * 30, "apply#3", '#' * 30)
    print(df.describe())
    print('=' * 30)
    # df2 = df.fillna(value=0)
    df2 = df
    df2 = df2.fillna(0)
    print(df.head())
    print('+' * 30)
    print(df2.head())
    print('=' * 30)
    df_nan = pd.isna(df)
    #print(pd.isna(df))

def apply_function4(df: pd.DataFrame):
    """ 
    https://sparkbyexamples.com/pandas/pandas-dataframe-mean-examples/
    """
    print('#' * 30, "apply#4", '#' * 30)
    for key in df.keys():
        df[key]    
    print('=' * 30)
    # df2 = df.fillna(value=0)
    df2 = df
    df2 = df2.fillna(0)
    print(df.head())
    print('+' * 30)
    print(df2.head())
    print('=' * 30)
    df_nan = pd.isna(df)
    #print(pd.isna(df))

def main():
    """ """
    df = pd.read_csv(f'./explore/cars.csv')
    # df = pd.read_csv('sample.csv', index_col=0)
    print(df.head(3))
    print('#' * 30)
    print(df.index)
    # print(df.columns)
    print(df.columns.to_list)
    print('#' * 30)
    print(df.describe())
    print('*' * 30)
    prices = df['price']
    # only numeric are displayed # col with NaN are removed
    print(df['price'].describe())  
    # best to select numeric
    # Syntax: dataFrameName.select_dtypes(include='number')
    ################## remove cols where NaN
    df_numerics_only = df.select_dtypes(include= np.number) # remove cols where NaN
    print(df_numerics_only)
    print('#' * 30)
    colnames_numerics_only = df.select_dtypes(include=np.number).columns.tolist()
    # #numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
    print(colnames_numerics_only)
    for key in df.keys():
        pass


if __name__ == "__main__":
    df = pd.read_csv(f'./explore/cars.csv')
    # main()
    print(df.mean(numeric_only=True))
    subsets(df)
    apply_function(df)
    apply_function2(df)
    apply_function4(df)

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