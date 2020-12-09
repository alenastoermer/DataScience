#!/usr/bin/env python
# coding: utf-8


import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data_preprocessed = pd.read_csv('Absenteeism_preprocessed.csv')

data_preprocessed.head()

# Regression, to give an idea which variables are important for the analysis
# Create classes for logistic regression
# Using median value of absenteeism time as cutoff line
# Thereby implicitly balancing the dataset
targets = np.where(data_preprocessed['Absenteeism Time in Hours'] >
                   data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)
data_preprocessed['Excessive Absenteeism'] = targets
data_preprocessed.head(20)

data_with_targets = data_preprocessed.drop(
    ['Absenteeism Time in Hours', 'Day of the Week', 'Daily Work Load Average', 'Distance to Work'], axis=1)

# Selecting, standardising inputs for regression
unscaled_inputs = data_with_targets.iloc[:, :-1]
unscaled_inputs.columns.values

columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Education']

columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]


# Custom scaler, to skip dummy variables
# To get a slightly more accurate, but harder to interpret model, 
# standardise all models with 
# absenteeism_scaler = StandardScaler()

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


absenteeism_scaler = CustomScaler(columns_to_scale)

absenteeism_scaler.fit(unscaled_inputs)

scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
# (Use transform() with this scaler for new data after model is deployed)

# Split input into training, test data; shuffle
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=20)

# Logistic Regression
reg = LogisticRegression()
reg.fit(x_train, y_train)

# Finding the intercept and coefficients, creating summary table
feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame(columns=['Feature Name'], data=feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table['Odds ratio'] = np.exp(summary_table.Coefficient)
summary_table.sort_values('Odds ratio', ascending=False)

# Interpretation: If coefficient is arount 0 or its odd ratio around 1,
# feature is probably not important lop
# In this case: Month value, Daily work load, distance to work, day of the week
# --> Go back to top to remove them via backwards elimination

# Since the baseline of our model is reason = 0, meaning no reason was given, the dummy vars mean
# the odds ratio of e.g. reason 3 = 22.100858 (poisoning) mean that someone is 22x more likely
# to be excessively absent when poisoned, than when no reason was given.
# Other, standardised vars like transportation expense are harder to interpret directly

reg.score(x_test, y_test)
# 0.75

# Find out probability of predicting a 0 or 1
predicted_proba = reg.predict_proba(x_test)
predicted_proba[:, 1]

# Pickle model
with open('model_absenteeism', 'wb') as file:
    pickle.dump(reg, file)

# Pickle scaler
with open('scaler_absenteeism', 'wb') as file:
    pickle.dump(absenteeism_scaler, file)
