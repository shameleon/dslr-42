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
    target = "Hogwarts House"
    feature1 = sys.argv[1]
    feature2 = sys.argv[2]
    df = pd.read_csv(f'./datasets/dataset_train.csv')
    sns.jointplot(data=df, x=feature1, y=feature2, hue=target)
    # for house in df['Hogwarts House'].unique():
    #     df[feature1]
    #     ax.scatter(x=df[feature1], y=df[feature2],
    #                 label=list(df['Hogwarts House'].unique()), alpha=0.3)

    # ax.set_xlabel(f'{feature1}', fontsize=12)
    # ax.set_ylabel(f'{feature2}', fontsize=12)
    # ax.set_title('Title')

    # ax.grid(True)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main()
    else:
        print("No arguments provided")

"""
Best Hand,Arithmancy,Astronomy,Herbology,Defense Against the Dark Arts,
Divination,Muggle Studies,Ancient Runes,History of Magic,Transfiguration,
Potions,Care of Magical Creatures,Charms,Flying
"""