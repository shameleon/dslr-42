import time
import numpy as np

"""
program:
      displays a  progress bar on the stdout
"""
COL_RESET = '\x1b[0m'
COL_ORANGE = '\x1b[38:5:208m'

def make_bar(i: int, epochs: int, scale_factor=20):
    full = int(epochs / scale_factor)
    progress = int((i + 1) / scale_factor)
    bar = ['Progress:', f'{int(100 * (i + 1) / epochs):>4}% ',
           COL_ORANGE, '[', '#' * progress,
           '.' * (full - progress), ']', COL_RESET]
    if full != progress:
        print(''.join(['{}'.format(x) for x in bar]), end='\r')
    else:
        print(''.join(['{}'.format(x) for x in bar]))


def my_iteration(epochs:int) -> None:
    """ """
    arr = np.zeros(10, dtype=float)
    perturbation = np.linspace(-0.1, 0.1, num=10)
    for i in range(epochs):
        np.random.shuffle(perturbation)
        arr += perturbation
        # bar = make_bar(i, epochs)
        time.sleep(0.05)
        n.zfill(3 if n < 0 else 2)
        print(''.join([f'|    {x:>3.2{2.2 + (x < 0)}f}  ' for x in arr])) #, end='\r')

"""
https://www.pylenin.com/blogs/python-width-precision/
 Let’s generalize 0.3f into a formula - x.yf

    x represents the minimum width or padding of the output string.
    y represents the maximum number of characters after the decimal.
    f symbolizes floating-point representation.
"""

def main():
    """  """
    my_iteration(10)

if __name__ == "__main__":
    main()

