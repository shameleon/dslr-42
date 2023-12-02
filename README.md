# dslr-42

## Datascience X Logistic Regression

![electronic hat](./graphics/bing-dslr-electroning-hat.jpg)

- [Intro](#intro)
- [Data understanding](#data-understanding)
- [Usage](#usage)
    - [describe.py](#describe-from-scratch)
    - [Plots](#plots)
    - [logreg_train.py](#training-dataset-with-logreg_trainpy)

- [Subject](#subject)
    - [Mandatory](#mandatory-part)
        - [Describe from scratch](#describe-from-scratch)
        - [Logistic regression training](#logistic-regression-training)
        - [Prediction](#prediction)
    - [Bonus](#bonus)
    - [Peer-Evaluation](#peer-evaluation)

- [Some helpful links](#some-helpful-links)

### Intro

The **DataScience x Logistic Regression** (`dslr`) is a *42's school* project, on the data branch of the *Holygraph*. As an initiation to *Machine Learning*, it consists of training a *multivariate logistic regression model*, to solve a classification problem.
This our solution `dslr` subject, done by a team of two, mostly in `Python` language.
`Python` functions or modules that are useful for statistics or machine learning, such as `scikit-learn`, are forbiden by *42*.
---

    The main objective is to recreate a ✨ magic Sorting Hat 🎓 ✨ to predict Hogwarts student houses.
    When [Harry Potter's universe](https://www.wizardingworld.com/) meets a Data scientist.

### Data understanding

The training dataset consists of 1600 students 🧙 caracteristics, with 17 features :

- Four Biographic features `First Name` `Last Name` `Birthday` `Best Hand`.

- A set of 13 wizard skills features being refered as : `Arithmancy` `Astronomy` `Herbology` `Defense Against the Dark Arts` `Divination` `Muggle Studies` `Ancient Runes` `History of Magic` `Transfiguration` `Potions` `Care of Magical Creatures` `Charms` `Flying`.

A model is trained, based on specific selected features, so that it can predict student's affiliation to one of the four 🏰 hogwart's houses
    🦅 `Gryffindor`
    🦡 `Hufflepuff`
    🐦‍⬛ `Ravenclaw`
    🐍 `Slytherin`

The targeted accuracy for predicting testing dataset should be above *98*%.

Scatter plots matrix visualization of students features. As preliminary work, we investigated the relationship between two variables taken two-by-two.
From there, we selected features that suits the best to train our model.

![Pair plot](./reports/pairplot.png)

---

### Usage

`make` to install the *virtual environment* with its *requirements*.

virtual environment `venv` activation:

```source venv/bin/activate```

then

``` (venv) ➜  dslr-42 git:(main) ✗ python describe.py ./datasets/dataset_train.csv ```

Entrypoints are at the root of the project :

|Program|Arguments|Action|
|---|---|---|
|`describe.py`|[dataset]|Describing a datasetwith statistics|
|`logreg_train.py`|[dataset]|Training logistic regression model from a training dataset|
|`logreg_test.py`|[dataset] [weights]|Testing logistic regression model with a testing dataset|
|`histogram.py`|[dataset] [feature]| plots a histogram for a given dataset feature|
|`scatter_plot.py`|[dataset] [feature_1] [feature_2]| Plots a scatter-plot for 2 given features|
|`pair_plot.py`|[dataset] | Plots a triangle-matrix of scatter-plots and distrbution for all dataset features|

Examples :

``` python ./dslr/logreg_train.py ./datasets/dataset_train.csv ```

to train the dataset.

``` python ./logreg_predict.py ./datasets/dataset_test.csv ./logistic_reg_model/gradient_descent_weights.csv ```

 to predict.

#### describe.py

`describe.py` mimics *pandas* library `describe()` function.
A data file must be provided as argument.

Describing the training dataset :

```python ./dslr/describe.py datasets/dataset_train.csv
```

Output:

```table
         Index Arithmancy Astronomy  ... Care of Magical Creatures   Charms   Flying
count  1600.00    1566.00   1568.00  ...                   1560.00  1600.00  1600.00
mean    799.50   49634.57     39.80  ...                     -0.05  -243.37    21.96
std     462.02   16679.81    520.30  ...                      0.97     8.78    97.63
min       0.00  -24370.00   -966.74  ...                     -3.31  -261.05  -181.47
25%     399.75   38511.50   -489.55  ...                     -0.67  -250.65   -41.87
50%     799.50   49013.50    260.29  ...                     -0.04  -244.87    -2.51
75%    1199.25   60811.25    524.77  ...                      0.59  -232.55    50.56
max    1599.00  104956.00   1016.21  ...                      3.06  -225.43   279.07
```

### Plots

Plots that are required

- Histogram

```python . histogram.py ./datasets/dataset_train.csv```

- Scatter Plot

```python scatter_plot.py ./datasets/dataset_train.csv```

- Pair plot Matrix

```python pair_plot.py ./datasets/dataset_train.csv```

Additional plots :

- Box plot `-b option`

```python dslr/plot_dataset.py -b ./datasets/dataset_train.csv```

- Joint plot `-j option`
Joint plot is a nice combination of scatter plot and density distribution.

```python dslr/plot_dataset.py ./datasets/dataset_train.csv```

Other plots in notebooks :

    multi-box plots and many heatmaps

- [1.0 notebook](./notebooks/1.0-jm-dataset-preview.ipynb)

- [2.0 notebook](./notebooks/2.0-jm-model-training.ipynb)

### Training dataset with logreg_train.py

```logreg_predict.[extension] ./datasets/dataset_train.csv [weights]```

## Subject

### Mandatory part

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

```shell
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


## Some helpful links

### Jupyter

Jupyter notebooks were used for dataset exploration.

### Directory Structure

Project directory structure was organized accordingly with the following guidelines.

[The Hitchhiker's Guide to Python - Structuring Your Project](https://docs.python-guide.org/writing/structure/)

[CookieCutter utility](https://drivendata.github.io/cookiecutter-data-science/#cookiecutter-data-science)

[How To Structure a Data Science Project: A Step-by-Step Guide](https://www.kdnuggets.com/2022/05/structure-data-science-project-stepbystep-guide.html)

### virtual environment

A Python [virtual environment](https://docs.python.org/3/library/venv.html) is installed and set up so that this project is self-contained, isolated from the system Python and from other projects virtual environments.
The virtual environment has its own Python Interpreter and dependencies as third-party libraries that are installed from `requirement.txt` file specifications. It avoids system pollution, dependency conflicts and optimizes reproducibility for a data science project. We used `virtualenv` tool for dependency management and project isolation. Instead of using `bash` script, we chose to exploit `Makefile` capabilities and readability for generic management tasks.

### Makefile and entrypoint

[Makefile: the secret weapon for ML project management](https://valohai.com/blog/makefile-ml-project-management/)

[Makefile - Make for Data Science](https://datasciencesouth.com/blog/make)

[setup.py script (french)](https://docs.python.org/fr/3/distutils/setupscript.html)

### setup.py

A setup.py file is a standard way in Python to specify how your project should be installed, packaged, and distributed. This file is used by tools like `setuptools` and `pip` to manage the installation process. The `setup()` function within `setup.py` is used to define various *metadata* about your project, such as its name, version, dependencies, and other details.
`python setup.py install` to install your project locally.

[setuptools](https://setuptools.pypa.io/en/latest/)

Having a setup.py becomes especially important if you plan to distribute your code as a Python package, whether through the Python Package Index (PyPI) or other distribution channels. It helps other developers easily install and use your project and allows tools like pip to manage dependencies.

### format, width, precision

[Precision](https://www.pylenin.com/blogs/python-width-precision/)

### pandas dataframes and np.arrays

[subsetting data](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html)
[Numpy hierachy](https://numpy.org/doc/stable/reference/arrays.scalars.html)

### for the describe.py part

[Argument parser = argparse](https://docs.python.org/3/library/argparse.html)
[Exceptions](https://docs.python.org/3/tutorial/errors.html)
[Pandas describe doc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html#)
[numpy statistics](https://numpy.org/doc/stable/reference/routines.statistics.html)
[numpy percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)
[Math and statistics online calculator](https://www.calculator.net/math-calculator.html)
[skweness and kurtosis](https://inside-machinelearning.com/skewness-et-kurtosis/)

[subset data](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html)

### for the logistic regression part

[Kaggle : logistic regression from scratch](https://www.kaggle.com/code/jagannathrk/logistic-regression-from-scratch-python)

### Plots

[constrained layout](https://matplotlib.org/stable/tutorials/intermediate/constrainedlayout_guide.html#sphx-glr-tutorials-intermediate-constrainedlayout-guide-py)

### testing

[unittest](https://docs.python.org/fr/3/library/unittest.html)
[unittest (in french)](https://gayerie.dev/docs/python/python3/unittest.html)
[unittest tutorial - openclassrooms](https://openclassrooms.com/fr/courses/7155841-testez-votre-projet-python/7414161-ajoutez-des-tests-avec-unittest)

#### Tests

Test runner chosen : `unittest` included in Python standard library.

`./dslr/tests/testDescribe.py` compares `DescriberClass` and pandas.describe()

`./dslr/tests/testUtilsMath.py` compares `utils.math.py` functions and numpy / pandas equivalent functions

#### describe.exe

An executable application could be built with
[pyinstaller](https://realpython.com/pyinstaller-python/)

- adding the entry-point script

```shell
pyinstaller ./dslr/describe.py
```

#### Icons

[Icons at mui.com](https://mui.com/material-ui/material-icon)
