
"""
Variables that are global to all dslr module
"""

""" dataset properties"""
target_label = 'Hogwarts House'
excluded_features = ["Arithmancy",
                     "Defense Against the Dark Arts",
                     "Care of Magical Creatures"]

""" Training dataset """
learning_rate = 0.01
epochs=100


""" Model """
model_dir = './logistic_reg_model/'
gradient_descent_weights = 'gradient_descent_weights.csv'

""" """
bonus = False
