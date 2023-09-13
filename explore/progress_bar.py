import time
import numpy as np

"""
program:
      displays a  progress bar on the stdout
"""

def my_iteration_bar(epochs:int) -> None:
    """ """
    for i in range(epochs):
        bar = ["[", "#" * int(i / 20),
        "." * int((epochs - i) /20), "]"]
        print('Progress:', *bar, end='\r')
        time.sleep(0.01)
    print('Progress:', *bar, end='\r')

def my_iteration_loop(epochs:int) -> None:
    """ """
    for i in range(epochs):
        print(f'Iteration: {i + 1}', end='\r')
        time.sleep(0.002)
    print(f'Iteration: {i + 1}')

def main():
    """  """
    my_iteration_loop(1000)
    time.sleep(0.5)
    my_iteration_bar(1000)

if __name__ == "__main__":
    main()

