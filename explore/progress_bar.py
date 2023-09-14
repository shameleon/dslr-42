import time
import numpy as np

"""
program:
      displays a  progress bar on the stdout
      progress is percentage of completed iterations
      Parameters: present iteration,
                    total iterations.
"""
COL_RESET = '\x1b[0m'
COL_ORANGE = '\x1b[38:5:208m'

def my_important_stuff():
    pass

def display_bar(i: int, epochs: int):
    bar_size = 50
    scale_factor = epochs / bar_size
    full = int(epochs / scale_factor)
    progress = int((i + 1) / scale_factor)
    bar = ['Progress:', f'{int(100 * (i + 1) / epochs):>4}% ',
           COL_ORANGE, '[', '#' * progress,
           '.' * (full - progress), ']', COL_RESET]
    if full != progress:
        print(''.join(['{}'.format(x) for x in bar]), end='\r')
    else:
        print(''.join(['{}'.format(x) for x in bar]))


def my_iteration_bar(epochs:int) -> None:
    """ """
    for i in range(epochs):
        my_important_stuff()
        display_bar(i, epochs)
        time.sleep(2 / epochs)


def main():
    """  """
    my_iteration_bar(800)

if __name__ == "__main__":
    main()

