import matplotlib.pyplot as plt
import os
import pandas as pd
import pydot

from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from subprocess import call

def calculate_accuracy(list_1, list_2):
    matches = 0
    for _ in range(len(list_1)):
        if list_1[_] == list_2[_]:
            matches += 1
    return (matches / len(list_1)) * 100

##########################
# Decision Tree Creation #
##########################

# Import data
userprofile = pd.read_csv(os.path.join('data', 'userprofile.csv'))
rating_final = pd.read_csv(os.path.join('data', 'rating_final.csv'))

# Picking useful features
full_dataset = pd.merge(rating_final, userprofile[[
    'userID', 
    'budget', 
    'activity',
    'smoker',
    'drink_level',
    'dress_preference',
    'ambience',
    'transport',
    'marital_status',
    'hijos',
    'birth_year',
    'interest',
    'personality',
    'religion',
    'color',
    'weight',
    'height']])

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

most_common_activity = features['activity'].value_counts().idxmax()
features['activity'] = features['activity'].replace({'?': most_common_activity})

most_common_smoker = features['smoker'].value_counts().idxmax()
features['smoker'] = features['smoker'].replace({'?' : most_common_smoker})

most_common_drink_level = features['drink_level'].value_counts().idxmax()
features['drink_level'] = features['drink_level'].replace({'?': most_common_drink_level})

most_common_dress_preference = features['dress_preference'].value_counts().idxmax()
features['dress_preference'] = features['dress_preference'].replace({'?': most_common_dress_preference})

most_common_ambience = features['ambience'].value_counts().idxmax()
features['ambience'] = features['ambience'].replace({'?': most_common_ambience})

most_common_transport = features['transport'].value_counts().idxmax()
features['transport'] = features['transport'].replace({'?': most_common_transport})

most_common_marital_status = features['marital_status'].value_counts().idxmax()
features['marital_status'] = features['marital_status'].replace({'?': most_common_marital_status})

most_common_hijos = features['hijos'].value_counts().idxmax()
features['hijos'] = features['hijos'].replace({'?': most_common_hijos})

most_common_interest = features['interest'].value_counts().idxmax()
features['interest'] = features['interest'].replace({'?': most_common_interest})

most_common_personality = features['personality'].value_counts().idxmax()
features['personality'] = features['personality'].replace({'?': most_common_personality})

most_common_religion = features['religion'].value_counts().idxmax()
features['religion'] = features['religion'].replace({'?': most_common_religion})

most_common_color = features['color'].value_counts().idxmax()
features['color'] = features['color'].replace({'?': most_common_color})

# OneHotEncoding of categorical data into continuous values
categorical_features = features.select_dtypes(include=[object])
onehot_features = pd.get_dummies(categorical_features)

# Merge OneHotEncoded values back into features set -- remove non-encoded values
features = pd.concat([features, onehot_features], axis=1).drop(list(features.select_dtypes(include=[object]).columns), axis=1)


for i in range(10):
    # Split data into train and test sets
    # NOTE: Train set builds our model, test set is tested against and used
    #       to determine accuracy of the model
    features, test_set, label, test_label = train_test_split(features, label, test_size=0.2)

    # Creating decision tree model from our features and label
    model = tree.DecisionTreeClassifier()
    model.fit(features, label)

    # Generate decision tree visualization
    tree.export_graphviz(model, out_file='budget_tree.dot', feature_names=features.columns[:])
    call(["dot", "-Tpng", "budget_tree.dot", "-o", "budget_tree.png"])

    ##########################
    # Performance Evaluation #
    ##########################

    test_results = model.predict(test_set)
    print(calculate_accuracy(test_results, test_label.values.T.tolist()))


