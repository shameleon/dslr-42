import argparse
import os

""" 
Histogram
Make a script called histogram.[extension] which displays a
histogram answering the next question :
Which Hogwarts course has a homogeneous score distribution
between all four houses?
"""


if __name__ == "__main__":
    """ """
    script_path = './dslr/plot_dataset.py'
    parser = argparse.ArgumentParser(prog='histogram.[ext]',
                                     description='histogram for a given feature of a dataset')
    parser.add_argument('filename')
    parser.add_argument('features', nargs=1, type=str)
    args = parser.parse_args()
    os.system(f'./venv/bin/python {script_path} {args.filename}'
              + f' -i {args.features[0]}')