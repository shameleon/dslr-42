import numpy as np
import pandas as pd
from utils import describe_stats as dum
import config


if __name__ == "__main__":
    df = pd.read_csv(f'./datasets/dataset_train.csv')
    df.drop(['Index'], inplace=True, axis=1)
    df = df.select_dtypes(include=np.number)
    df_means = df.agg(lambda feat: dum.mean(feat), axis=0)
    df_stds = df.agg(lambda feat: dum.std(feat), axis=0)
    df_stats = pd.DataFrame({'mean_x': df_means,
                             'std_x': df_stds}
                           ).T
    print(df_stats)

