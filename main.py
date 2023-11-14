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

