import unittest

class DescribeTesting(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        
    def setup(self):
        print("Unit testing for describe.py")

if __name__ == '__main__':
    unittest.main()