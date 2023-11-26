import argparse
import os

"""
    Describes the data of a dataset file

    executes ./dlsr/describe.py script
    takes one argument 'filename' .csv file
"""


if __name__ == "__main__":
    """ """
    script_path = './dslr/describe.py'
    parser = argparse.ArgumentParser(prog='describe.[ext]',
                                     description='describe data of a dataset',
                                     epilog='verbose mode for options')
    parser.add_argument('filename')
    args = parser.parse_args()
    os.system(f'./venv/bin/python {script_path} {args.filename}')
