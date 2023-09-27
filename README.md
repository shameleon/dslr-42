# dslr-42 : Datascience X Logistic Regression

---

Machine Learning project from 42 school, holygraph outside circle.

It consists of training a ***multivariate logistic regression model** to solve a classification problem.

The main objective is to recreate a magic Sorting Hat to predict Hogwarts student houses.

---

## Subject

### Mandatory

```describe.py``` program describes a dataset, has similar behavior to ```nympy.describe()```

```multi-classifier using a logistic regression one-vs-all```
logreg_train.[extension] dataset_train.csv

gradient descent to minimize the error

generates a file
containing the weight

 logreg_predict.[extension] dataset_train.csv [weights]

 generate a prediction file houses.csv formatted exactly as follows:
```
$> cat houses.csv
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
[...] 
```

### Bonus

• Add more fields for describe.[extension] 
    such as [skweness and kurtosis](https://inside-machinelearning.com/skewness-et-kurtosis/)
• Implement a stochastic gradient descent
• Implement other optimization algorithms (Batch GD/mini-batch GD/ you name

### Peer-Evaluation

Answers will be evaluated using accuracy score of the Scikit-Learn library. Professor
McGonagall agrees that your algorithm is comparable to the Sorting Hat only if it has a
minimum precision of 98% .

## Usage

    ```Makefile``` creates a virtual environment. 
    ```make``` command, then
    ```shell
    source /venv/bin/activate
    ```

## Best

(Kaggle : logistic regression from scratch)[https://www.kaggle.com/code/jagannathrk/logistic-regression-from-scratch-python]

## Toolkit

### Describe like Pandas does

(Pandas describe() doc)[https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html#]


### Pandas

subset data https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html

Numpy hierachy https://numpy.org/doc/stable/reference/arrays.scalars.html

### Some helpful links

[Argument parser = argparse](https://docs.python.org/3/library/argparse.html)
[Exceptions](https://docs.python.org/3/tutorial/errors.html)
[numpy percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)