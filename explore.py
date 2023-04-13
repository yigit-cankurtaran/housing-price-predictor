from sklearn.datasets import fetch_california_housing
import pandas as pd

# loading the data
housing = fetch_california_housing()

# export the data into a pandas dataframe
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# exploring the data
print(f'The shape of the data is: {df.shape}')
print(df.head())
print(df.describe())
# df.describe() gives us the mean, std, min, max, 25%, 50%, 75% of the data

print(df.dtypes)
# df.dtypes gives us the data types of each column
# in this dataset, all the columns are float64

# checking for missing values
print(df.isnull().sum())
# no missing values
