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
import mpl_toolkits.mplot3d

#%% load data

# Load data without "Delay" column (attribute)
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

df = df.drop("CDR", axis=1) # Remove 'Hand' column

print(df.head()) # Check if it worked
# %% Extract the raw data
attributeNames = np.asarray(df.columns)
print(attributeNames)
X = df.values
N, M = X.shape
print(N,M)