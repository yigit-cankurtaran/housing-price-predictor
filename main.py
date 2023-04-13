from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading the data
housing = fetch_california_housing()

# export the data into a pandas dataframe
df = pd.DataFrame(housing.data, columns=housing.feature_names)
print(f'The shape of the data is: {df.shape}')
print(df.head())
