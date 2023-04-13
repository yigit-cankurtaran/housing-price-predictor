from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# loading the data
housing = fetch_california_housing()

# export the data into a pandas dataframe
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# feature engineering
print(df.columns)
df['Households'] = df['Population'] / df['AveOccup']
print(df['Households'].head())
df['RoomsPerHousehold'] = df['AveRooms'] / df['Households']
df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
df['PopulationPerHousehold'] = df['Population'] / df['Households']
df['IncomePerHousehold'] = df['MedInc'] / df['Households']
df['IncomePerOccupant'] = df['MedInc'] / df['AveOccup']
# ^ raised the accuracy by 3%
print(df.columns)
#  the number increases because we added new columns
# feature engineering helps us improve the model's performance
#  because we are adding new columns that are more useful than the original ones

# we can also drop columns that are not useful


# scaling the data
scaler = StandardScaler()
X = scaler.fit_transform(df)
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# regularize the data
model = Ridge()
model.fit(X_train, y_train)

# evaluate the model
print(f'The model\'s R^2 score is: {model.score(X_test, y_test)}')

# visualize the model's predictions
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
# plt.show()

# visualize the model's residuals
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
# plt.show()

# visualize the model's residuals' distribution
plt.hist(residuals, bins=150)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
# plt.show()
