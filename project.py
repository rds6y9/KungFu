import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error
from sklearn import tree
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder
from subprocess import call
import os
import pydot

## Command-line configuration
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', -1)

## Original csv's, don't touch after loading, create duplicates
accepts = pd.read_csv(os.path.join('data', 'chefmozaccepts.csv'))
cuisine = pd.read_csv(os.path.join('data', 'chefmozcuisine.csv'))
hours = pd.read_csv(os.path.join('data', 'chefmozhours4.csv'))
parking = pd.read_csv(os.path.join('data', 'chefmozparking.csv'))
geo = pd.read_csv(os.path.join('data', 'geoplaces2.csv')) 
usercuisine = pd.read_csv(os.path.join('data', 'usercuisine.csv'))
payment = pd.read_csv(os.path.join('data', 'userpayment.csv'))
profile = pd.read_csv(os.path.join('data', 'userprofile.csv'))
rating = pd.read_csv(os.path.join('data', 'rating_final.csv'))

## Rating and Profile cleaning / merging
full_dataset = pd.merge(rating, profile, on=['userID']) 

## Geoplaces cleaning / merging 
# geo_dataset = geo
# geo_dataset = geo_dataset.rename(columns={'latitude': 'restaurant_latitude', 'longitude': 'restaurant_longitude'})

# full_dataset = pd.merge(full_dataset, geo_dataset, on=['placeID'])

## OneHotEncoding -- used to analyze categorical data
smoker_converted = pd.get_dummies(full_dataset, columns=['smoker'])
drink_level_converted = pd.get_dummies(smoker_converted, columns=['drink_level'])
dress_preference_converted = pd.get_dummies(drink_level_converted, columns=['dress_preference'])
ambience_converted = pd.get_dummies(dress_preference_converted, columns=['ambience'])
transport_converted = pd.get_dummies(ambience_converted, columns=['transport'])
marital_status_converted = pd.get_dummies(transport_converted, columns=['marital_status'])
budget_converted = pd.get_dummies(marital_status_converted ,columns=['budget'])
color_converted = pd.get_dummies(budget_converted ,columns=['color'])
activity_converted = pd.get_dummies(color_converted ,columns=['activity'])
religion_converted = pd.get_dummies(activity_converted, columns=['religion'])
personality_converted = pd.get_dummies(religion_converted, columns=['personality'])
interest_converted = pd.get_dummies(personality_converted, columns=['interest'])
hijos_converted = pd.get_dummies(interest_converted, columns=['hijos'])

full_dataset = hijos_converted

## Decision Tree Classifiers

# CART Comparison

model = tree.DecisionTreeClassifier()
# Separate features from label (in our case, rating is the label)
data = full_dataset.drop(columns=['userID', 'placeID'])
target = full_dataset['rating']

model.fit(data, target)
tree.export_graphviz(model, out_file='user_profile_tree.dot')
call(["dot", "-Tpng", "user_profile_tree.dot", "-o", "user_profile_tree.png"])
print(full_dataset)


# testResults = model.predict()
# print(testResults)


## Correlation calculations
# print(full_dataset['rating'].corr(full_dataset['smoker']))
