# Import all we need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
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
#%%

# Load data without "Delay" column (attribute)
df = pd.read_csv("../mri-and-alzheimers/oasis_cross-sectional.csv", dtype=None, delimiter=',', encoding=None, usecols = range(11))
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
# %% Extract the raw data
attributeNames = np.asarray(df.columns)
print(attributeNames)
X = df.values
N, M = X.shape
print(N,M)
#%%
#plt.figure(dpi=150) # Spyder plot size
#fig = plt.figure(figsize = (width_fullsize,height)) # Full page width
plt.figure(figsize = (width, height)) # Single column width
plt.boxplot(X)
plt.xticks(range(1,M+1), attributeNames, rotation=45)
plt.grid()
#plt.savefig('Figures/boxplot.pgf') # Save the file
plt.show()
#%%
NEW_DATA = zscore(X,ddof=1)

#plt.figure(dpi=150) # Spyder plot size
#fig = plt.figure(figsize = (width_fullsize,height)) # Full page width
plt.figure(figsize = (width, height)) # Single column width
plt.boxplot(NEW_DATA)
plt.xticks(range(1,M+1), attributeNames, rotation=45)
plt.grid()
#plt.savefig('Figures/boxplot_Normal.pgf') # Save the file
plt.show()
