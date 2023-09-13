import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

"""
hhttps://seaborn.pydata.org/examples/spreadsheet_heatmap.html
"""
def main():
    file = f'logistic_reg_model/gradient_descent_weights.csv'
    df_weights = pd.read_csv(file)
    print(df_weights.columns)
    # flights = df_weights.pivot("month", "year", "passengers")
    # f, ax = plt.subplots(figsize=(9, 6))
    # sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)
    # plt.show()

if __name__ == "__main__":
    main()
