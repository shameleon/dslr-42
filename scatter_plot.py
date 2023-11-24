import argparse
import os

"""
V.2.2 Scatter plot
Make a script called scatter_plot.[extension] which displays
a scatter plot answering the next question :
What are the two features that are similar ?
answer : sns_scatter('Astronomy', 'Defense Against the Dark Arts')
"""


if __name__ == "__main__":
    """ """
    description = 'scatter plot for a given feature of a dataset'
    script_path = './dslr/plot_dataset.py'
    parser = argparse.ArgumentParser(prog='scatter_plot.py',
                                     description=description)
    parser.add_argument('filename')
    parser.add_argument('features', nargs=2, type=str)
    args = parser.parse_args()
    os.system(f'./venv/bin/python {script_path} {args.filename}'
              + f' -s {args.features[0]} {args.features[1]}')
