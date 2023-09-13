import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

"""
https://foundations.projectpythia.org/core/matplotlib/matplotlib-basics.html
https://foundations.projectpythia.org/core/matplotlib/annotations-colorbars-layouts.html
"""
def main():
    file = f'logistic_reg_model/gradient_descent_weights.csv'
    df_weights = pd.read_csv(file)
    print("+", df_weights.head())
    
if __name__ == "__main__":
    main()
