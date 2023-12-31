"""
Datascience X Logistic Regression
dslr/config.py

Variables that are global to all dslr module
"""

""" datasets properties"""
target_label = 'Hogwarts House'

""" Training dataset parameters """
excluded_features = ["Arithmancy",
                     "Defense Against the Dark Arts",
                     "Care of Magical Creatures"
                     ]

""" Training dataset parameters """
learning_rate = 0.1
epochs = 1000
algorithm = 'Gradient Descent'

""" Model directory """
model_dir = './logreg_model/'
standardization_params = 'standardization_params.csv'
gradient_descent_weights = 'weights.csv'

""" Test model """
test_truth = 'dataset_truth.csv'

""" Bomus """
bonus = False

"""
all_features = ['Arithmancy', 'Astronomy', 'Herbology',
                'Defense Against the Dark Arts,Divination',
                'Muggle Studies', 'Ancient Runes',
                'History of Magic', 'Transfiguration',
                'Potions', 'Care of Magical Creatures',
                'Charms', 'Flying']
"""
