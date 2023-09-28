# Dslr  : Datascience X Logistic Regression

---

Machine Learning project at 42 school, from the holygraph outside circle.

Dslr consists of training a ***multivariate logistic regression model** to solve a classification problem.

---

## Subject

### Synopsis

The main objective is to recreate a magic Sorting Hat to predict Hogwarts student houses. When Harry Potter universe meet a Data sciencist.

### Mandatory

#### Describe from scratch

A ```describe.py``` program to describe the dataset, that behaves as ```nympy.describe()```. It is forbidden to use any function that makes the job,
like: count, mean, std, min, max, percentile, etc...

#### Logistic regression training

```multi-classifier using a logistic regression one-vs-all```
logreg_train.[extension] dataset_train.csv

Gradient descent algoritm to minimize the error

Generates a file containing the model weights.

Usage :

```shell
logreg_predict.[extension] dataset_train.csv [weights]
```

#### Prediction

Predict from '.datasets/dataset_test.csv' and generate a prediction file `houses.csv`` formatted exactly as follows:

```
$> cat houses.csv
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
[...] 
```

### Bonus

• Add more fields for describe.py
• Implement a stochastic gradient descent
• Implement other optimization algorithms (Batch GD/mini-batch GD/ you name

### Peer-Evaluation

Answers will be evaluated using accuracy score of the Scikit-Learn library. Professor
McGonagall agrees that your algorithm is comparable to the Sorting Hat only if it has a
minimum precision of 98% .

## My solution to Dslr

### venv

Virtual environment is created by a `Makefile`.
In a terminal, ```make``` command, then

```shell
source /venv/bin/activate
```

### Toolkit and testing

## Some helpful links

### Fluency with pandas dataframes and np.arrays

[subset data](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html)
[Numpy hierachy](https://numpy.org/doc/stable/reference/arrays.scalars.html)

### for the describe.py par

[Argument parser = argparse](https://docs.python.org/3/library/argparse.html)
[Exceptions](https://docs.python.org/3/tutorial/errors.html)
[Pandas describe doc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html#)
[numpy percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)
[skweness and kurtosis](https://inside-machinelearning.com/skewness-et-kurtosis/)

[subset data](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html)

### for the logistic regression part

[Kaggle : logistic regression from scratch](https://www.kaggle.com/code/jagannathrk/logistic-regression-from-scratch-python)

### Plots
 
[constrained layout](https://matplotlib.org/stable/tutorials/intermediate/constrainedlayout_guide.html#sphx-glr-tutorials-intermediate-constrainedlayout-guide-py)