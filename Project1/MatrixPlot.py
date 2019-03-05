# Import all we need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

df = pd.read_csv("../mri-and-alzheimers/oasis_cross-sectional.csv", dtype=None, delimiter=',', encoding=None, usecols = range(11))
df = df.drop("ID", axis=1) # Remove 'ID' column
df = df.drop("Hand", axis=1) # Remove 'Hand' column
print(df.columns)
#df = df.drop("CDR", axis=1)
df = df.dropna() # Drop all rows with any NaN cells

classLabels = df['CDR'].unique() # Returns the unique CDR vales: 0,0.5,1,2
y = np.asarray(df['CDR']) # Class vector
C = len(classLabels)

print(df.head())

df = df.drop("CDR", axis=1) # Remove 'Hand' column
#df = df.drop("M/F", axis=1)
#df = df.drop("Educ", axis=1)
#df = df.drop("SES", axis=1)

print(df.head()) # Check if it worked
# %% Extract the raw data
attributeNames = np.asarray(df.columns)
print(attributeNames)
X = df.values
N, M = X.shape
print(N,M)
# %%
plt.figure(figsize=(12,10))
color_array = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
for m1 in range(M):
    for m2 in range(M):
        plt.subplot(M, M, m1*M + m2 + 1)
        if m1 != 0 and m2 != 0:
            corr = np.corrcoef(X[:,m2].astype(np.float32), X[:,m1].astype(np.float32))
        i = 0
        for c in classLabels:
            class_mask = (y==c)
            plt.plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.', color=color_array[i])
            if m1==M-1:
                plt.xlabel(attributeNames[m2])
            else:
                plt.xticks([])
            if m2==0:
                plt.ylabel(attributeNames[m1])
            else:
                plt.yticks([])
            i += 1
#        plt.title(np.round(corr[0,1],2))
        if m1 != m2:
            plt.title("$cor$ = %1.2f" % np.round(corr[0,1],2))
plt.legend(classLabels)
#plt.savefig('Figures/pairplot.png')
plt.show()