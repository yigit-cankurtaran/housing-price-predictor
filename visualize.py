from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# loading the data
housing = fetch_california_housing()

# export the data into a pandas dataframe
df = pd.DataFrame(housing.data, columns=housing.feature_names)

#  visualize the distributions of each column
df.hist(bins=50, figsize=(20, 15))
plt.show()

# visualize the correlation between the columns with a heatmap
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

# visualize the relationship between median income and average bedrooms
plt.scatter(df['MedInc'], df['AveBedrms'])
plt.xlabel('Median Income')
plt.ylabel('Average Bedrooms')
plt.show()
