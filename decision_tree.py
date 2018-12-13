import matplotlib.pyplot as plt
import os
import pandas as pd
import pydot

from sklearn import tree
from sklearn import preprocessing
from subprocess import call


## Minimal Decision Tree Creation

# Import data
userprofile = pd.read_csv(os.path.join('data', 'userprofile.csv'))
rating_final = pd.read_csv(os.path.join('data', 'rating_final.csv'))

# Picking useful features
full_dataset = pd.merge(rating_final, userprofile[['userID', 'budget']])

# Split data into features (determining factors) and label (rating)
# Remove userID and placeID as they shouldnt be part of the decision tree
# These values are located at index 0 and 1 of the dataset
features = full_dataset[list(full_dataset.columns[2:])]
features = features.drop(columns=['rating', 'food_rating', 'service_rating'])

label = full_dataset['rating']

# Handling missing values in features. Different columns must be handled differently.
# Our standard approach will be to fill in missing values with mean or middle values.
# Leaving missing values will make the decision tree perform poorly.
features['budget'] = features['budget'].replace({'?': 'medium'})

# OneHotEncoding of categorical data into continuous values
X = features.select_dtypes(include=[object])
X = pd.get_dummies(X)

# Merge OneHotEncoded values back into features set
features = pd.concat([features, X], axis=1).drop(['budget'], axis=1)

# Creating decision tree model from our features and label
model = tree.DecisionTreeClassifier()
model.fit(features, label)

tree.export_graphviz(model, out_file='budget_tree.dot', feature_names=features.columns[:])
call(["dot", "-Tpng", "budget_tree.dot", "-o", "budget_tree.png"])
