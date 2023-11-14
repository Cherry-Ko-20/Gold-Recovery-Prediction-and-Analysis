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

# #Check that recovery is calculated correctly 
train[train['rougher.output.recovery'].notna()]['rougher.output.recovery']
print(train['rougher.output.recovery'])

#  the top 10 non-NaN values from the 'rougher.output.recovery' column 
train[train['rougher.output.recovery'].notna()]['rougher.output.recovery'].sort_values(ascending=False).head(10)
print(train)

# Define a function to calculate recovery
def calculate_recovery(row):
    try:
        if pd.isna(row['rougher.input.feed_au']) or pd.isna(row['rougher.output.concentrate_au']) or pd.isna(row['rougher.output.tail_au']):
            return np.nan  # Return NaN when any of the input values is missing
        elif row['rougher.input.feed_au'] == 0 or row['rougher.output.concentrate_au'] == 0:
            return 0
        else:
            return (row['rougher.output.concentrate_au'] *
                    (row['rougher.input.feed_au'] - row['rougher.output.tail_au'])) / \
                    (row['rougher.input.feed_au'] *
                    (row['rougher.output.concentrate_au'] - row['rougher.output.tail_au'])) * 100
    except ZeroDivisionError:
        return 0

# Calculate recovery using the defined function
train['recovery_calc'] = train.apply(calculate_recovery, axis=1)


print(train['recovery_calc'])

# Drop rows with missing values in the calculated recovery
train = train.dropna(subset=['recovery_calc', 'rougher.output.recovery'])

# Calculate MAE using the aligned data
mae = mean_absolute_error(train['rougher.output.recovery'], train['recovery_calc'])
print("Mean Absolute Error (MAE):", mae)

# Show column differences between dataframes
train_columns = train.columns
test_columns = test.columns
column_differences = train_columns.difference(test_columns)
print(column_differences)

print(source.isna().sum())
print(train.isnull().sum())
print(test.isnull().sum())

# First, drop rows with missing target values
source.dropna(subset=['rougher.output.recovery', 'final.output.recovery'], inplace=True)

# Then, fill missing values with 0 for the remaining columns
source_update = source.fillna(method='ffill')
source_update

#creating train_updated using same rows as in train from source since missing values filled 
train_updated = source[source['date'].isin(train['date'])]
train_updated.describe()

#creating test_updated using same rows as in train from source since missing values filled 
test_updated = source[source['date'].isin(test['date'])]

#removing column_differences from test_updated 
test_updated = test_updated[test_updated.columns.difference(column_differences)]
print(test_updated)

#dropping any rows that still have missing values 
source.dropna(how='any', inplace=True, axis=0)
train_updated.dropna(how='any', inplace=True, axis=0)
test_updated.dropna(how='any', inplace=True, axis=0)

# Create subplots to visualize concentration changes
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

# List of metals
metals = ['au', 'ag', 'pb']

# Iterate through the metals and plot their concentration changes
for i, metal in enumerate(metals):
    train[f'rougher.input.feed_{metal}'].plot.hist(ax=axes[i], label='Raw Feed', legend=True)
    train[f'rougher.output.concentrate_{metal}'].plot.hist(ax=axes[i], label='Rougher Concentrate', legend=True)
    train[f'final.output.concentrate_{metal}'].plot.hist(ax=axes[i], label='Final Concentrate', legend=True)

    axes[i].set_title(f'Concentration of {metal}')
    axes[i].set_xlabel('Sample Index')
    axes[i].set_ylabel('Concentration')

plt.tight_layout()
plt.show()

# Calculate total concentrations at different stages
train['total_concentration_raw'] = train.filter(like='rougher.input.feed_', axis=1).sum(axis=1)
train['total_concentration_rougher'] = train.filter(like='rougher.output.concentrate_', axis=1).sum(axis=1)
train['total_concentration_final'] = train.filter(like='final.output.concentrate_', axis=1).sum(axis=1)

# Examine summary statistics
print("Summary Statistics of Total Concentrations:")
print(train[['total_concentration_raw', 'total_concentration_rougher', 'total_concentration_final']].describe())

# Create subplots to visualize the histograms of total concentrations
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

# List of total concentration variables
total_concentration_variables = ['total_concentration_raw', 'total_concentration_rougher', 'total_concentration_final']

# Iterate through the total concentration variables and plot histograms
for i, concentration_variable in enumerate(total_concentration_variables):
    train[concentration_variable].plot.hist(ax=axes[i], bins=30, color='skyblue', edgecolor='black')
    axes[i].set_title(f'Histogram of {concentration_variable}')
    axes[i].set_xlabel('Total Concentration')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create normalized histograms for the training and test sets
train_size = train['rougher.input.feed_size']
test_size = test['rougher.input.feed_size']

# Define the number of bins for the histograms
bins = 50

# Plot the normalized histograms
ax.hist(train_size, bins=bins, density=True, alpha=0.7, label='Training Set')
ax.hist(test_size, bins=bins, density=True, alpha=0.7, label='Test Set')

# Set axis labels and title
ax.set_xlabel('Feed Particle Size')
ax.set_ylabel('Density')
ax.set_title('Normalized Feed Particle Size Distribution Comparison')

# Add a legend to distinguish the datasets
ax.legend()

# Show the plot
plt.show()

#removing columns from training data that are not in test data and date column which shouldn't be predictive 
train_updated_features = train_updated[train_updated.columns.difference(column_differences)]
train_updated_features.drop('date', axis=1, inplace=True)
train_updated_features

#creating train target
train_updated_target = train_updated[['rougher.output.recovery', 'final.output.recovery']].copy()
train_updated_target

def smape(target, prediction): 
    return (1/len(target)) * sum(abs(prediction - target) / ((abs(target) + abs(prediction)) / 2) * 100)

def final_smape(target, prediction):
    rougher = smape(target[:, 0], prediction[:, 0])
    final = smape(target[:, 1], prediction[:, 1])
    return rougher * 0.25 + final * 0.75

#silencing warnings 
import sys
import warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")

    #setting scorer
smape_score = make_scorer(final_smape)

#saving values from training features and target
X_train = train_updated_features.values
y_train = train_updated_target.values

#training linearregression and evaluating using cross validation 
lr = LinearRegression().fit(X_train, y_train)
scores_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring=smape_score)
print("Mean sMAPE:", np.nanmean(scores_lr))
scores_lr

#training randomforestregressor and evaluating using cross validation 
for depth in range(1,6): 
    r_forest = RandomForestRegressor(n_estimators = 50, max_depth=depth, random_state=
                                12345).fit(X_train, y_train)
    scores_r_forest = cross_val_score(r_forest, X_train, y_train, cv=5, scoring=smape_score)
    print("Depth: ", depth, "Mean sMAPE:", np.nanmean(scores_r_forest))
    print("Cross validation scores: ", scores_r_forest)
    print()



# Define a simpler parameter grid to start with
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
}

rf_model = RandomForestRegressor(random_state=12345)
smape_score = make_scorer(final_smape, greater_is_better=False)

# Initialize GridSearchCV with the model, parameter grid, and scoring metric
grid_search = GridSearchCV(rf_model, param_grid, scoring=smape_score, n_jobs=-1, cv=5, verbose=4)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

print("Mean sMAPE for the best Random Forest model:", np.abs(grid_search.best_score_))
