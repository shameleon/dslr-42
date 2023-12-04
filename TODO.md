# dslr TODO.md

### Issues

- [ ] issue : Display live cost during training - might slow down
- [ ] issue #cheating : describe is used for testing in DescriberClass main and testDescribe.py
- [ ] issue : Scale pairplot matrix to screen - Dell clusters
- [ ] issue : Display more dataset details during training
- [ ] issue : os.system is used, pass eveything to [os.subprocess](https://docs.python.org/fr/3/library/subprocess.html#module-subprocess) [replace with subprocess](https://docs.python.org/fr/3/library/subprocess.html#subprocess-replacements)

### Reading

- [Model Evaluation](https://www.geeksforgeeks.org/machine-learning-model-evaluation/?ref=ml_lbp)
- [mutlilabel ranking metrics](https://www.geeksforgeeks.org/multilabel-ranking-metrics-label-ranking-average-precision-ml/)
- [skewness-et-kurtosis](https://inside-machinelearning.com/skewness-et-kurtosis/)
- [Logistic Regression with Stochastic Gradient Descent](https://www.kaggle.com/code/marissafernandes/logistic-regression-sgd-in-python-from-scratch)
- [Gradient descent algorithms](https://realpython.com/gradient-descent-algorithm-python/)

### Todo

- [ ] describe.py output saved to file *BonusFeature*
- [ ] summary for the training as in [odds-ratio-logistic-regressionodds-ratio-logistic-regression](https://mmuratarat.github.io/2019-09-05/odds-ratio-logistic-regression)
- [ ] add plot logreg-predict.py
- [ ] *Bonus* : Stochastic Gradient Descent (SGD)
- [ ] *Bonus* : Minibatch (SGD)

### In Progress
- [ ] Plug in a progress bar to training           *BonusFeature*
- [ ] *Bonus* for describe.py
  - [ ] -b option for bonus
  - [X] Count NaNs
  - [ ] unique, top, freq
  - [ ] Skewness, Kurtosis, No. of 3 Sigma Outliers 
- [ ] *Bonus* : choose relevent model metrics and write to file 
  - [X] Accuracy
  - [ ] Precision, recall, F1 score
  - [ ] Confusion matrix - plot heatmap
  - [ ] MAE, MSE, RMSE, MAPE
  - [ ] LRAP Label Ranking average precision
- [x] Model Evaluation with `sklearn`` module
  - [ ] Jupyter notebook
    - [ ] Cross-validation
    - [ ] The Receiver Operating Characteristic (ROC) curve

### Testing


### Done ✓

- [x] directory structure
  - [x] dslr module
  - [x] subshell all scripts
- [x] setup virtual environment
- [x] describing dataset
  - [x] math abd stats tools module dslr_stats
  - [x] Describe class
  - [x] describe.py at root
  - [x] Many Tests
- [x] Data Visualization
  - [x] Histogram
  - [x] Scatter plot
  - [x] Pair plot
- [x] Data Visualization
  - [x] more plots (boxplt, jointplot)             *BonusFeature*
- [x] Jupyter notebooks
  - [x] explain pipeline and logistic regression
  - [x] plots in jupyter : multiplots              *BonusFeature*
- [x] Group plots in a class
- [x] Refactor logreg-train.py
  - [x] add loss function plot                     *BonusFeature*
   - [x] add weights heat-map function plot        *BonusFeature*
- [x] Refactor logreg-predict.py
  - [x] add weights argument to logreg
  - [x] handle real output for both datasets

### Results so far

`99.0%` prediction accuracy on the testing dataset with a 10-feature based model.
