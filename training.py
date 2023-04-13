from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# loading the data
housing = fetch_california_housing()

# export the data into a pandas dataframe
df = pd.DataFrame(housing.data, columns=housing.feature_names)

#  preprocess the dataset, split it into train and test sets
X = df
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
