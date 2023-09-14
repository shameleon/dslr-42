#!/usr/bin/env python
import pandas as pd
import sys

""" program arguments parsing """


def main():
    try:
        df = pd.read_csv(f'{sys.argv[1]}')
        print(df.head())
    except (IndexError):
        print(f'Usage: \'{sys.argv[1]}\' <filename.csv>')
    except (FileNotFoundError, IsADirectoryError, pd.errors.ParserError):
        print(f'Error: \'{sys.argv[1]}\' is not a .csv')


if __name__ == "__main__":
    main()