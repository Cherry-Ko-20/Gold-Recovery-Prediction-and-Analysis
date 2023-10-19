import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV

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


train['rougher.output.recovery'] = train ['rougher.output.recovery'].fillna(0)
train['recovery_calc'] = train['recovery_calc'].fillna(0)
print(train)

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

# Handle Missing Values
train.dropna(inplace=True)
test.dropna(inplace=True)

# Analyze the data 
# Create subplots to visualize concentration changes
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# List of metals
metals = ['au', 'ag', 'pb']

# Iterate through the metals and plot their concentration changes
for i, metal in enumerate(metals):
    # Train data
    train[f'rougher.output.concentrate_{metal}'].plot(ax=axes[i], label='Raw Feed', legend=True)
    train[f'primary_cleaner.output.concentrate_{metal}'].plot(ax=axes[i], label='Rougher Concentrate', legend=True)
    train[f'final.output.concentrate_{metal}'].plot(ax=axes[i], label='Final Concentrate', legend=True)

    axes[i].set_title(f'Concentration of {metal}')
    axes[i].set_xlabel('Sample Index')
    axes[i].set_ylabel('Concentration')

plt.tight_layout()
plt.show()


# Compare the feed particle size distributions in the training set and the test set
plt.figure(figsize=(10, 6))

plt.hist(train['rougher.input.feed_size'], bins=50, label='Training Set', alpha=0.7)
plt.hist(test['rougher.input.feed_size'], bins=50, label='Test Set', alpha=0.7)
plt.xlabel('Feed Particle Size')
plt.ylabel('Frequency')
plt.title('Feed Particle Size Distribution Comparison')
plt.legend()

plt.show()

# Calculate the total concentrations at different stages
train['total_concentration_raw'] = train.filter(like='raw', axis=1).sum(axis=1)
train['total_concentration_rougher'] = train.filter(like='rougher', axis=1).sum(axis=1)
train['total_concentration_final'] = train.filter(like='final', axis=1).sum(axis=1)

# Examine summary statistics
print("Summary Statistics of Total Concentrations:")
print(train[['total_concentration_raw', 'total_concentration_rougher', 'total_concentration_final']].describe())

