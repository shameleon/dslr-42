import time
import numpy as np

"""
program:
    displays a 1D numpy array to stdout
    view is refreshed while the array is updated 
"""
COL_RESET = '\x1b[0m'
COL_ORANGE = '\x1b[38:5:208m'

def show_array_progress(title: str, arr:np.ndarray, refresh=True) -> None:
    """
    displays a 1D numpy array to stdout
    as a refreshed table row with constant padding.
    This is preceeded with a title, describing the array.
    a refresh mode True will place end='\r' in the print function.

    As for processing of array content, with join() method
    padding with f-string formula x.yf, where 
    x represents the minimum width or padding of the output string.
    y represents the maximum number of characters after the decimal.
    f symbolizes floating-point representation.
    zero and positive numbers are filled with an extra space char
    to compensate for negative numbers minus sign for negative 
    An end='\r' to print refreshed lines while refresh parameter is True
    and end='\n' when refresh parameter is False (last display view)
    """
    weights_as_row = ''.join([f'| {bool(x >= 0) * " "}{x:>3.2f} ' for x in arr])
    head = f'{COL_ORANGE}{title}{COL_RESET}'
    end_with =  (1 - refresh) * '\n' + refresh * '\r'
    print(head, weights_as_row, end=end_with)

def my_iteration(epochs: int) -> None:
    """
    A weights array is randomly updated through np.shuffle method,
    joined as an f-string. """
    arr = np.zeros(10, dtype=float)
    perturbation = np.linspace(-0.1, 0.1, num=10)
    for i in range(epochs):
        show_array_progress("weights", arr)
        np.random.shuffle(perturbation)
        arr += perturbation
        time.sleep(0.1)
    show_array_progress("weights", arr, False)


def main():
    """  """
    my_iteration(10)

if __name__ == "__main__":
    main()

