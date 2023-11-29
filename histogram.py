import argparse
import os

"""
Histogram
Make a script called histogram.[extension] which displays a
histogram answering the next question :
Which Hogwarts course has a homogeneous score distribution
between all four houses?
'Arithmancy'
'Care of Magical Creatures'
"""


if __name__ == "__main__":
    """call plot_dataset script with the -i option for histogram
    $ python histogram.py ./datasets/dataset_train.csv 'Charms'
    """
    description = 'histogram for a given feature of a dataset'
    script_path = './dslr/plot_dataset.py'
    parser = argparse.ArgumentParser(prog='histogram.py',
                                     description=description)
    parser.add_argument('filename')
    parser.add_argument('feature', nargs=1, type=str)
    args = parser.parse_args()
    print("Histogram plot for", os.path.split(args.filename)[1])
    print('feature :', args.feature[0])
    os.system(f'./venv/bin/python {script_path} {args.filename}'
              + f' -i "{args.feature[0]}"')
