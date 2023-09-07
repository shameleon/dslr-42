import sys

""" program arguments parsing """

def put_args():
    for i in range(1, len(sys.argv)):
        print(sys.argv[i])

if __name__ == "__main__":
    if (len(sys.argv) >= 2):
        put_args()