
subject :
```multi-classifier using a logistic regression one-vs-all```
[https://www.youtube.com/watch?app=desktop&v=oYVdzBjkiZ4]

Three types of Logistic Regression:
1) Binomial: Where target variable is one of two classes
2) Multinomial: Where the target variable has three or more possible classes
3) Ordinal: Where the target variables have ordered categories

Binary classification vs. Multi-class classification
Multi-class classification : 
    One-One 
    One-vs-all

Model 1: A or not A
Model 2: B or not B
Model 3: C or not C

## One-vs-all 
One-Vs-All Classification is a method of multi-class classification. It can be broken down by splitting up the multi-class classification problem into multiple binary classifier models. For k class labels present in the dataset, k binary classifiers are needed in One-vs-All multi-class classification.

MLE 
()[https://towardsdatascience.com/multi-class-classification-one-vs-all-one-vs-one-94daed32a87b]
best formulas
()[https://www.cs.rice.edu/~as143/COMP642_Spring22/Scribes/Lect5]

One-vs-all classification breaks down k classes present in our dataset D into k binary
classifier models that aims to classify a data point as either part of the current class ki or as
part of all other classes. Each model can discriminate the ith class with everything else

## Best
(Kaggle : logistoic regression from scratch)[https://www.kaggle.com/code/jagannathrk/logistic-regression-from-scratch-python]

equations and def in python
sigmoid
loss 

(Logistic regression from scratch)[https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac]
best formulas
(Building the Logistic Regression Function)[https://beckernick.github.io/logistic-regression-from-scratch/]

[https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/]

equations and def in python

[https://www.datacamp.com/tutorial/understanding-logistic-regression-python]

Maximum Likelihood Estimation Vs. Least Square Method

The MLE is a "likelihood" maximization method, while OLS is a distance-minimizing approximation method. Maximizing the likelihood function determines the parameters that are most likely to produce the observed data. From a statistical point of view, MLE sets the mean and variance as parameters in determining the specific parametric values for a given model. This set of parameters can be used for predicting the data needed in a normal distribution.

Ordinary Least squares estimates are computed by fitting a regression line on given data points that has the minimum sum of the squared deviations (least square error). Both are used to estimate the parameters of a linear regression model. MLE assumes a joint probability mass function, while OLS doesn't require any stochastic assumptions for minimizing distance.

[https://www.stat4decision.com/fr/faire-une-regression-logistique-avec-python/]

[https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8]

Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.
Logistic Regression Assumptions

    Binary logistic regression requires the dependent variable to be binary.
    For a binary regression, the factor level 1 of the dependent variable should represent the desired outcome.
    Only the meaningful variables should be included.
    The independent variables should be independent of each other. That is, the model should have little or no multicollinearity.
    The independent variables are linearly related to the log odds.
    Logistic regression requires quite large sample sizes.


Standardization

    les données ne sont pas standardisées, leur interprétation dépendra de l’ordre de grandeur des échelles des variables.

## swarm plots
[https://www.kaggle.com/code/alexisbcook/scatter-plots]


## missing data

[https://bookdown.org/egarpor/PM-UC3M/app-nas.html]

Use complete cases. This is the simplest solution and can be achieved by restricting the analysis to the set of fully-observed observations. 


## probabilities
[https://stats.oarc.ucla.edu/r/dae/multinomial-logistic-regression/]

[https://bookdown.org/egarpor/PM-UC3M/app-ext-multinomialreg.html]



## correlation matrix
