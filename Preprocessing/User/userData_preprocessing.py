import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Adding column names
column_headers = ['userId', 'gameName', 'isPlay', 'playTime', 'temp']

# Loading data
df = pd.read_csv('Dataset/steam-200k.csv', encoding='utf-8', names=column_headers)

# Setting the index
df.set_index('userId', inplace=True)

print('[head]\n', df.head())
print('\n[tail]\n', df.tail())

# Counting the number of null values
# There is no null values in this dataset
print('\n[number of null]\n', df.isnull().sum())

# Removing empty rows with no significance
df = df.drop('temp', axis=1)

# Removing games that were only purchased
df = df[~(df['isPlay'] == "purchase")]

# Removing outliers based on standard deviation
df = df[~(np.abs(df.iloc[:, 2] - df.iloc[:, 2].mean()) > (3 * df.iloc[:, 2].std()))].fillna(0)

# Counting the number of times each playtime occurs
y = df['playTime'].value_counts()
y = y.sort_index()

print('\n[number of playtime]\n', y)
print('\n[head after preprocessing]\n', df.head())
print('\n[preprocessing]\n', df)
df.to_csv('final_user_data.csv')