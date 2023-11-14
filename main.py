import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error 
import matplotlib.pyplot as plt
from scipy import stats as st
from sklearn.linear_model import LinearRegression
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from math import sqrt
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.model_selection import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.dummy import DummyRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.utils import shuffle

train= pd.read_csv('files/gold_recovery_train.csv')
source=pd.read_csv('files/gold_recovery_full.csv')
test=pd.read_csv('files/gold_recovery_test.csv')
print(train.info())
print(source.info())
print(test.info())

missing_values= train.isnull().sum()
print(missing_values)

# #Check that recovery is calculated correctly 
train[train['rougher.output.recovery'].notna()]['rougher.output.recovery']
print(train['rougher.output.recovery'])

#  the top 10 non-NaN values from the 'rougher.output.recovery' column 
top_10_recovery_values =train[train['rougher.output.recovery'].notna()]['rougher.output.recovery'].sort_values(ascending=False).head(10)
print(top_10_recovery_values)

# Calculate recovery
# Calculate recovery with handling division by zero
def calculate_recovery(row):
    if row['rougher.input.feed_au'] == 0 or row['rougher.output.concentrate_au'] == 0:
        return 0
    else:
        return (row['rougher.output.concentrate_au'] *
                (row['rougher.input.feed_au'] - row['rougher.output.tail_au'])) / \
                (row['rougher.input.feed_au'] *
                (row['rougher.output.concentrate_au'] - row['rougher.output.tail_au'])) * 100

train['recovery_calc'] = train.apply(calculate_recovery, axis=1)
print(train['recovery_calc'])


# Fill NaN values with 0
train['rougher.output.recovery'] = train['rougher.output.recovery'].fillna(method='ffill')
train['recovery_calc'] = train['recovery_calc'].fillna(method='ffill')

# Calculate MAE
mae = mean_absolute_error(train['rougher.output.recovery'], train['recovery_calc'].notna())
print("Mean Absolute Error (MAE):", mae)

# Analyze the features not available in the test set

missing_columns = set(train.columns) - set(test.columns)
missing_columns_list = list(missing_columns)
missing_columns_types = train[missing_columns_list].dtypes
print("Missing columns:", missing_columns)
print("Data types of missing columns:", missing_columns_types) 


# Perform Data Processing 

# Fill missing values in all datasets with 0
train = train.fillna(method='ffill')
test = test.fillna(method='ffill')
source = source.fillna(method='ffill')


# Create subplots to visualize concentration changes
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# List of metals
metals = ['au', 'ag', 'pb']

# Iterate through the metals and plot their concentration changes
for i, metal in enumerate(metals):
    train[f'rougher.output.concentrate_{metal}'].plot(ax=axes[i], label='Raw Feed', legend=True)
    train[f'primary_cleaner.output.concentrate_{metal}'].plot(ax=axes[i], label='Rougher Concentrate', legend=True)
    train[f'final.output.concentrate_{metal}'].plot(ax=axes[i], label='Final Concentrate', legend=True)

    axes[i].set_title(f'Concentration of {metal}')
    axes[i].set_xlabel('Sample Index')
    axes[i].set_ylabel('Concentration')

plt.tight_layout()
plt.show()


# Calculate the total concentrations at different stages
train['total_concentration_raw'] = train.filter(like='raw', axis=1).sum(axis=1)
train['total_concentration_rougher'] = train.filter(like='rougher', axis=1).sum(axis=1)
train['total_concentration_final'] = train.filter(like='final', axis=1).sum(axis=1)

# Examine summary statistics
print("Summary Statistics of Total Concentrations:")
print(train[['total_concentration_raw', 'total_concentration_rougher', 'total_concentration_final']].describe())

# Compare the feed particle size distributions in the training set and the test set
plt.figure(figsize=(10, 6))

plt.hist(train['rougher.input.feed_size'], bins=50, label='Training Set', alpha=0.7)
plt.hist(test['rougher.input.feed_size'], bins=50, label='Test Set', alpha=0.7)
plt.xlabel('Feed Particle Size')
plt.ylabel('Frequency')
plt.title('Feed Particle Size Distribution Comparison')
plt.legend()

plt.show()

# Define a function to calculate sMAPE
def calculate_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    smape = 100 * np.mean(diff)
    return smape


# Create a custom scorer for sMAPE
smape_scorer = make_scorer(calculate_smape, greater_is_better=False)

# Define the features and target variable
features = train.drop(['date', 'final.output.recovery'], axis=1)
target = train['final.output.recovery']

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  
rf_model.fit(X_train, y_train)

# Perform cross-validation to evaluate the model
smape_scores = -cross_val_score(rf_model, X_train, y_train, cv=3, scoring=smape_scorer)

# Calculate the mean sMAPE score
mean_smape = smape_scores.mean()
print(f"Mean sMAPE on cross-validation: {mean_smape:.2f}%")


from sklearn.linear_model import LinearRegression

# Create a Linear Regression model
lr_model = LinearRegression()

# Train the Linear Regression model
lr_model.fit(X_train, y_train)

# Perform cross-validation for Linear Regression
lr_smape_scores = -cross_val_score(lr_model, X_train, y_train, cv=3, scoring=smape_scorer)

# Calculate the mean sMAPE score for Linear Regression
mean_lr_smape = lr_smape_scores.mean()
print(f"Mean sMAPE for Linear Regression on cross-validation: {mean_lr_smape:.2f}%")


# Define the features in the test set
test_features = test.drop(['date', 'final.output.recovery'], axis=1)

# Predict 'final.output.recovery' values in the test set
test_predictions = rf_model.predict(test_features)

# Calculate the final sMAPE value for the test dataset
final_sMAPE = calculate_smape(test['final.output.recovery'], test_predictions)
print(f"Final sMAPE on the test dataset: {final_sMAPE:.2f}%")