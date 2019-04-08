#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:28:35 2019

@author: Yssubhi
"""
#%% Import libs

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mpl_toolkits.mplot3d
from scipy.stats import zscore
from scipy.linalg import svd
import seaborn as sns


#%% load data

# Load data without "Delay" column (attribute) //// insert own path!
df = pd.read_csv("/Users/Yssubhi/Downloads/02450/mri-and-alzheimers/oasis_cross-sectional.csv", dtype=None, delimiter=',', encoding=None, usecols = range(11))
# Remove 'ID' column
df = df.drop("ID", axis=1)
# Remove 'Hand' column
df = df.drop("Hand", axis=1)
print(df.columns)
#df = df.drop("CDR", axis=1)
# Drop all rows with any NaN cells
df = df.dropna()
print(df['ASF'].describe())

print(df.head())
# Number of male participants
print("Male:", (df['M/F'] == 'M').sum())
# Number of female participants
print("Female:", (df['M/F'] == 'F').sum())

# Since a gender (M/F) given as a character either M or F is useless,
# designate M as 0, and female as 1.
gender_mask = {"M" : 0, "F" : 1}
df['M/F'] = df['M/F'].map(gender_mask)


# Returns the unique CDR vales: 0,0.5,1,2
classLabels = df['CDR'].unique()
 # Class vector
y = np.asarray(df['CDR'])
C = len(classLabels)

# Remove 'Hand' column
df = df.drop("CDR", axis=1)

 # Check if it worked
print(df.head())
# %% Extract the raw data
attributeNames = np.asarray(df.columns)
print(attributeNames)
X = df.values
N, M = X.shape
print(N,M)

# for fun.. print X
print(X)

# %% standard derivations

X_z = zscore(X)
# plt.plot(X_z)
# print(X_z)

# %% One out of K encoding of gender attributes (M/F into M & F)

print(attributeNames)
gender = np.array(X[:, 0], dtype=int).T
# K number of "columns" in the one out of K encoding
K = gender.max() + 1

# make empty two column N row array
gender_encoding = np.zeros((gender.size, K))
# Map the correct values
gender_encoding[np.arange(gender.size), gender] = 1

# Concatante to end of old data and remove the first column (M/F)
X = np.concatenate( (X[:, 1:M], gender_encoding), axis=1)

# Remap attributes
attributeNames = np.append(attributeNames[1:M], ["M", "F"])
print(X[0,:])
# Sanity check, should start with 'age', end with 'F'
print(attributeNames)

# Should now be 216 x 9 (1 attribute more due to encoding)
N, M = X.shape
print(N, 'x', M)

# %%
