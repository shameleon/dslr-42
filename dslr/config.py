
"""
Variables that are global to all dslr module
"""

""" datasets properties"""
target_label = 'Hogwarts House'

""" Training dataset parameters """
excluded_features = ["Arithmancy",
                     "Defense Against the Dark Arts",
                     "Care of Magical Creatures"]
learning_rate = 0.1
epochs=1000
algorithm = 'Gradient Descent'

""" Model directory """
model_dir = './logistic_reg_model/'
standardization_params = 'standardization_params.csv'
gradient_descent_weights = 'gradient_descent_weights.csv'

""" Test model """
test_truth = 'dataset_truth.csv'

""" Bomus """
bonus = False
