#!/usr/bin/env python3
# Import all we need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

print(plt.rcParams['axes.prop_cycle'].by_key()['color']) # Default pyplot colors
#%% Setup plots for latex integration
import matplotlib as mpl
width = 86 * 0.0393700787
width_fullsize = 172 * 0.0393700787
height = 3.2
lw = 1.3

pgf_with_latex = {
    "pgf.texsystem": "pdflatex",         # use Xelatex which is TTF font aware
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",             # use serif rather than sans-serif
    "font.serif": "Times New Roman",             # use 'Ubuntu' as the standard font
    "font.sans-serif": [],
    "font.monospace": "Ubuntu Mono",    # use Ubuntu mono if we have mono
    "axes.labelsize": 12,               # LaTeX default is 10pt font.
    "font.size": 12,
    "legend.fontsize": 12,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "text.latex.unicode": True,
    "pgf.preamble": [
     r"\usepackage{amsmath}",
     ]
}

mpl.rcParams.update(pgf_with_latex)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#%% Load data


#df = pd.read_csv("mri-and-alzheimers/oasis_cross-sectional.csv", dtype=None, delimiter=',', encoding=None, usecols = range(11))
df = pd.read_csv("../mri-and-alzheimers/oasis_cross-sectional.csv", dtype=None, delimiter=',', encoding=None, usecols = range(11))
df = df.drop("ID", axis=1) # Remove 'ID' column
df = df.drop("Hand", axis=1) # Remove 'Hand' column
df = df.drop("CDR", axis=1)
df = df.drop("M/F", axis=1)
df = df.dropna() # Drop all rows with any NaN cells
#%%
attributeNames = np.asarray(df.columns)
print(attributeNames)
X = df.values
N, M = X.shape
print(N,M)

#%%

#plt.figure(dpi=150)
plt.figure(figsize = (width,height * 0.8)) # single column width

i = 1
for m1 in range(2):
    print("m1:", m1)
    for m2 in range(3):
        print("m2:", m2)
        plt.subplot(2, 3, i)
        plt.hist(X[:, i - 1])
        plt.xlabel(attributeNames[i - 1])
        plt.grid()
        i += 1
plt.savefig('Figures/histogram.pgf')
plt.show()

#%% standardised boxplot
#
#
# THIS ONE IS NOT USED. REFER TO boxplots.py
#
#

df = df.drop("ID", axis=1) # Remove 'ID' column
df = df.drop("Hand", axis=1) # Remove 'Hand' column
print(df.columns)
#df = df.drop("CDR", axis=1)
df = df.dropna() # Drop all rows with any NaN cells

print(df.head())
print("Male:", (df['M/F'] == 'M').sum()) # Number of male participants
print("Female:", (df['M/F'] == 'F').sum()) # Number of female participants

# Since a gender (M/F) given as a character either M or F is useless,
# designate M as 0, and female as 1.
gender_mask = {"M" : 0, "F" : 1}
df['M/F'] = df['M/F'].map(gender_mask)



classLabels = df['CDR'].unique() # Returns the unique CDR vales: 0,0.5,1,2
y = np.asarray(df['CDR']) # Class vector
C = len(classLabels)

df = df.drop("CDR", axis=1) # Remove 'Hand' column

print(df.head()) # Check if it worked
print(df['eTIV'].describe())

attributeNames = np.asarray(df.columns)
print(attributeNames)
X = df.values
N, M = X.shape
print(N,M)


#figure(dpi=150)
#title('Boxplot of Attributes: Standardized')
#boxplot(zscore(X, ddof=1))
#xticks(range(1,M+1), attributeNames, rotation=45)



