import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
train['recovery_calc'] = (train['rougher.output.concentrate_au'] *
                               (train['rougher.input.feed_au'] - train['rougher.output.tail_au'])) / \
                              (train['rougher.input.feed_au'] *
                               (train['rougher.output.concentrate_au'] - train['rougher.output.tail_au'])) * 100
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



