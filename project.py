import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import os

accepts = pd.read_csv(os.path.join('data', 'chefmozaccepts.csv'))
cuisine = pd.read_csv(os.path.join('data', 'chefmozcuisine.csv'))
hours = pd.read_csv(os.path.join('data', 'chefmozhours4.csv'))
parking = pd.read_csv(os.path.join('data', 'chefmozparking.csv'))
geo = pd.read_csv(os.path.join('data', 'geoplaces2.csv')) 
usercuisine = pd.read_csv(os.path.join('data', 'usercuisine.csv'))
payment = pd.read_csv(os.path.join('data', 'userpayment.csv'))
profile = pd.read_csv(os.path.join('data', 'userprofile.csv'))
rating = pd.read_csv(os.path.join('data', 'rating_final.csv'))