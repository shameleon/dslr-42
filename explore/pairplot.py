import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(f'../datasets/dataset_train.csv')
df.head()
# sns.pairplot(df[13:], hue="Hogwarts House", diag_kind="hist")