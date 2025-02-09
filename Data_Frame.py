# -*- coding: utf-8 -*-
"""
Spyder Editor

This script processes and analyzes data from the BDD100K dataset, focusing on autonomous driving scenes.
"""

import pandas as pd  # Importing pandas library for data manipulation

# Load JSON file into a DataFrame (bdd100k dataset containing labeled driving scenes)
df_train = pd.read_json('/Users/groy/Downloads/archive/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json')

df = df_train  # Creating a copy of the original dataset

df_train = df_train.head(500)  # Limiting the dataset to the first 500 rows for analysis

# =============================================================================
# Extracting 'attributes' column (which contains nested data) into a separate DataFrame
attributes_df = pd.json_normalize(df_train['attributes'])

# Removing 'attributes' column from df_train and merging extracted attributes back
# This makes it easier to work with the data

df_train = df_train.drop(columns=['attributes']).join(attributes_df)

# Dropping columns that are not required for analysis
# 'labels' and 'timestamp' are removed as they might not be necessary for this study
df_train = df_train.drop(columns=['labels'])
df_train = df_train.drop(columns=['timestamp'])

# Loading another dataset that contains car count information
df_train_500 = pd.read_csv('/Users/groy/Downloads/Project/Car_Count.csv')

# --------------------------
# Processing the full dataset (not just the first 500 rows) in a similar manner

# Extracting attributes from the full dataset
df_attributes = pd.json_normalize(df['attributes'])
df = df.drop(columns=['attributes']).join(df_attributes)

# Dropping unnecessary columns from the full dataset
df = df.drop(columns=['labels'])
df = df.drop(columns=['timestamp'])

# Merging the processed dataset with the car count dataset on 'name' column
# 'name' is assumed to be a unique identifier for each image

df_merged = pd.merge(df, df_train_500, on="name", how="inner")

# Selecting relevant columns for analysis
data = df_merged[['name', 'weather', 'scene', 'timeofday', 'Value']]
print(data.head())  # Displaying first few rows of processed data

# ---------------------------
# Calculating the average car count ('Value') for different time periods in the dataset
average_values = data.groupby('timeofday', as_index=False)['Value'].mean()
print(average_values)  # Displaying the results

# Calculating the average car count based on different driving scenes
average_values1 = data.groupby('scene', as_index=False)['Value'].mean()
print(average_values1)

# Calculating the average car count based on different weather conditions
average_values2 = data.groupby('weather', as_index=False)['Value'].mean()
print(average_values2)

# Saving the processed data to a CSV file for future use
data.to_csv('file1.csv')
