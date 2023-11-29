import argparse
import os

"""
Pair plot matrix : Plot pairwise relationships in a dataset.
V.2.3 Pair plot
Make a script called pair_plot.[extension] which displays
a pair plot or scatter plot matrix
(according to the library that you are using).
From this visualization, what features are you going
to use for your logistic regression?
see dslr/config.py for excluded features
"""

if __name__ == "__main__":
    """ """
    description = 'Pairplot matrix for all features of a dataset'
    script_path = './dslr/plot_dataset.py'
    parser = argparse.ArgumentParser(prog='pair_plot.py',
                                     description=description)
    parser.add_argument('filename')
    args = parser.parse_args()
    os.system(f'./venv/bin/python {script_path} {args.filename} -p')
