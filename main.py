import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error 
import matplotlib.pyplot as plt

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


# Calculate recovery
train['recovery_calc'] = (train['rougher.output.concentrate_au'] *
                               (train['rougher.input.feed_au'] - train['rougher.output.tail_au'])) / \
                              (train['rougher.input.feed_au'] *
                               (train['rougher.output.concentrate_au'] - train['rougher.output.tail_au'])) * 100
print(train['recovery_calc'])

