#!/usr/bin/env python
# coding: utf-8

import pandas as pd

raw_csv_data = pd.read_csv('Absenteeism_data.csv')
# Make working copy of data
df = raw_csv_data.copy()

# Get summary of dataframe, e.g. missing values
display(df)
df.info()

# Drop id column
df = df.drop(['ID'], axis=1)

# Create dummy variables for categorical data
reason_columns = pd.get_dummies(df['Reason for Absence'])
# Check if exactly one reason was given in each column
reason_columns['check'] = reason_columns.sum(axis=1)
reason_columns['check'].sum(axis=0)
reason_columns = reason_columns.drop(['check'], axis=1)

# Drop first column to avoid multicollinearity
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)

# Group reasons for absence, replace in df
reason_columns.loc[:, '1':'14']
reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)

df = df.drop(['Reason for Absence'], axis=1)
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)

# Rename new columns
df.columns.values
column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                'Daily Work Load Average', 'Body Mass Index', 'Education',
                'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
df.columns = column_names

# Reorder columns
column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense',
                          'Distance to Work', 'Age',
                          'Daily Work Load Average', 'Body Mass Index', 'Education',
                          'Children', 'Pets', 'Absenteeism Time in Hours']
df = df[column_names_reordered]

# Create a checkpoint in jupyter
df_reason_mod = df.copy()

# Change date type from string to timestamp
df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format='%d/%m/%Y')

# Extract Month
list_months = []

for i in range(df_reason_mod.shape[0]):
    list_months.append(df_reason_mod['Date'][i].month)

df_reason_mod['Month Value'] = list_months


# Extract weekday
def date_to_weekday(date_value):
    return date_value.weekday()


df_reason_mod['Day of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)

# Clean up dataframe
df_reason_mod = df_reason_mod.drop(['Date'], axis=1)
df_reason_mod.head()
df_reason_mod.columns.values

column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Day of the Week', 'Month Value',
                          'Transportation Expense', 'Distance to Work', 'Age',
                          'Daily Work Load Average', 'Body Mass Index', 'Education',
                          'Children', 'Pets', 'Absenteeism Time in Hours']
df_reason_mod = df_reason_mod[column_names_reordered]

df_reason_date_mod = df_reason_mod.copy()

# Create dummy var for education
df_reason_date_mod['Education'].unique()
df_reason_date_mod['Education'].value_counts()

# Map vars, combine all higher education
df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})

df_preprocessed = df_reason_date_mod.copy()
df_preprocessed.head(10)

df_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)
