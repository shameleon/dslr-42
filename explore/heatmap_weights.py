import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

"""
https://foundations.projectpythia.org/core/matplotlib/matplotlib-basics.html
https://foundations.projectpythia.org/core/matplotlib/annotations-colorbars-layouts.html
"""
def main():
    file = f'logistic_reg_model/gradient_descent_weights.csv'
    df_weights = pd.read_csv(file)
    df_weights.set_index('Unnamed: 0', inplace=True)
    print(df_weights.head(11))
    houses = df_weights.columns[1:].to_list()
    features = df_weights['Unnamed: 0'].to_list()
    # print(df_weights[:,1:])
    # sns.heatmap(df_weights.columns[1:].to_numpy())
    # plt.show()


if __name__ == "__main__":
    main()
