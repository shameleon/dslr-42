from setuptools import setup, find_packages

"""
name: The name of the project.
version: The version number of the project.
packages:
    Use find_packages() to automatically discover
    and include all Python packages in the project.
install_requires:
    A list of dependencies that are needed by the project.
    When someone installs your project using pip,
    these dependencies will be installed automatically.
entry_points: This is optional. If your project includes
 command-line scripts, you can use entry_points to define them.
"""

setup(
    name='dslr',
    version='0.1',
    packages=find_packages(),
    author='jmouaike',
    author_email='john.mouaike@gmail.com',
    description=("A machine learning 42-school project,"
                 "multivariate logistic regression"),
    license="BSD",
    keywords="42 dataset machine learning dataset logistic regression",
    long_description=('README'),
    install_requires=[
        'pandas',
        'matplotlib',
        'numpy',
        'seaborn'
        ],
    entry_points=''' ''',
)
