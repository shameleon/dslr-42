# dslr-42 : Datascience X Logistic Regression

---

Machine Learning project for 42 school

Train a logistic regression model to solve classification problem

Recreate a magic Sorting Hat to predict Hogwarts student houses.
---

## subject

### Mandatory

```multi-classifier using a logistic regression one-vs-all```
logreg_train.[extension] dataset_train.csv

gradient descent to minimize the error

generates a file
containing the weight

 logreg_predict.[extension] dataset_train.csv [weights]

 generate a prediction file houses.csv formatted exactly as follows:
$> cat houses.csv
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
[...]

### Bonus

• Add more fields for describe.[extension]
• Implement a stochastic gradient descent
• Implement other optimization algorithms (Batch GD/mini-batch GD/ you name

### Peer-Evaluation

Answers will be evaluated using accuracy score of the Scikit-Learn library. Professor
McGonagall agrees that your algorithm is comparable to the Sorting Hat only if it has a
minimum precision of 98% .

## Best

(Kaggle : logistic regression from scratch)[https://www.kaggle.com/code/jagannathrk/logistic-regression-from-scratch-python]

## Toolkit

### Describe like Pandas does

(Pandas describe() doc)[https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html#]


## Modules

https://docs.python.org/3/library/argparse.html

### Pandas

subset data https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html

Numpy hierachy https://numpy.org/doc/stable/reference/arrays.scalars.html
