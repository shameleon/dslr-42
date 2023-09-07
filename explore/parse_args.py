import argparse
import pandas as pd

""" program arguments parsing 
    usage: prog.py [-h] [--sum] N [N ...]
    Process some integers.

    positional arguments:
    N           an integer for the accumulator

    options:
    -h, --help  show this help message and exit
    --sum       sum the integers (default: find the max)
    """

def main():
    """ https://docs.python.org/3/library/argparse.html """
    # parser = argparse.ArgumentParser(description='Reads a .csv file')
    # parser.add_argument('-d', '--debug', help='Debugging output', action='store_true')
    # parser.add_argument('csvfile', type=argparse.FileType('r'), help='Input csv file')
    # args = parser.parse_args()

    # parser.add_argument('-h', '--help')
    # parser.add_argument('', )
    # parser.print_help()
    df = pd.read_csv(f'./explore/cars.csv')
    print(df.head())

if __name__ == "__main__":
    main()


"""
https://www.tutorialspoint.com/python/python_command_line_arguments.htm
"""