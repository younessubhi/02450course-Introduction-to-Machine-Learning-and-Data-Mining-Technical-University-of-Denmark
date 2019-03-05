# Import all we need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
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
print(df['ASF'].describe())

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
# %% Extract the raw data
attributeNames = np.asarray(df.columns)
print(attributeNames)
X = df.values
N, M = X.shape
print(N,M)

# %% One out of K encoding of M/F
print(attributeNames)
gender = np.array(X[:, 0], dtype=int).T
K = gender.max() + 1 # K number of "columns" in the one out of K encoding

gender_encoding = np.zeros((gender.size, K)) # make empty two column N row array
gender_encoding[np.arange(gender.size), gender] = 1 # Map the correct values

X = np.concatenate( (X[:, 1:M], gender_encoding), axis=1) # Concatante to
# end of old data and remove the first column (M/F)

attributeNames = np.append(attributeNames[1:M], ["M", "F"]) # Remap attributes
print(X[0,:])
print(attributeNames) # Sanity check, should start with 'age', end with 'F'

N, M = X.shape
print(N,M) # Should now be 216 x 10 (1 attribute more due to encoding)
# %% Having one-out-of-K encoded (or one-hot) the gender, the data
# needs to be normalized. Since 'eTIV' is so much larger
# than everything else, we normalize with the standard deviation as well

# If we only normalize with mean, there is still a size difference between
# the largest and smallest absolute value of 15200.
X_norm = X - np.ones((N,1))*X.mean(axis=0)
print(np.abs(X_norm[0]).max() / np.abs(X_norm[0]).min())

# Reducing furthermore subtracting the standard deviation reduces
# the factor down to 3400
X_norm = (X - np.ones((N,1)) * X.mean(axis=0)) / X.std(axis=0)
print(np.abs(X_norm[0]).max() / np.abs(X_norm[0]).min())

# %% As the data is now set, we can do a singular value decomposition (SVD)

U,S,V = svd(X_norm,full_matrices=False)

# Variance explained:
rho = (S*S) / (S*S).sum() 
print(S)
print("Sigma matrix:", np.round(S,2))

print("V:", np.round(V,2))

print(rho)

threshold = 0.9

# Plot variance explained
plt.figure(figsize = (width, height))
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
#plt.xticks([1,2,3,4,5,6,7,8,9, 10])
plt.xticks(np.linspace(1,M,M))
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
#plt.savefig('Figures/Variance_explained.pgf')
plt.show()

#%% PCA Component Coefficients
print(V[:,0])
print(len(attributeNames))
pcs = [0,1,2]
#pcs = [7,8,9]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
f = plt.figure(figsize = (width_fullsize,height))
#f = plt.figure(dpi=300) # To see shit on the own computer
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)


locs, labels = plt.xticks(r+bw, attributeNames)
#plt.setp(labels, rotation=90)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs, loc='upper right')
plt.grid()
#plt.savefig('Figures/PCA_Coefficients.pgf')
plt.show()

#%% Plot the first two principal components in 2d.
v1 = V[:,0] # PC1
v2 = V[:,1] # PC2
v3 = V[:,3] # PC3
print("PC1:", v1)
print("PC2:", v2)
print("PC3:", v3)

print(np.dot(v1,v2)) # Check for orthogonality (0 = linear independence)
origin = [0], [0] # Vector origin point

fig, ax = plt.subplots(dpi=150)
plt.quiver(*origin, v1[0:2], v2[0:2], color=['r','b','g'])

custom_lines = [Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='b', lw=4)]
ax.legend(custom_lines, ['PC1', 'PC2'])

plt.tight_layout()
plt.show()


#%% Transform the data into PC domain and plot 2d.
B = X@V[:,0:3]

#plt.figure(figsize = (width,height))
plt.figure(dpi=150)
color_array = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
i = 0
for c in classLabels:
    class_mask = y == c
    plt.scatter(B[class_mask,0], B[class_mask,1], label=c, color=color_array[i], alpha=.75)
    i += 1
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.legend()
#plt.legend(title='CDR', bbox_to_anchor=(0., 1.02, 1, .102), loc=3, ncol=4, mode="expand", borderaxespad=0)
#plt.savefig('Figures/2dprojection.pgf', bbox_inches="tight")
plt.show()

print(np.corrcoef(B[:,0], B[:,1]))
#%% Plot 3d


#fig = plt.figure(dpi=150)
fig = plt.figure(figsize = (width_fullsize,height))
ax = fig.add_subplot(111, projection='3d')
i = 0
for c in classLabels:
    class_mask = y == c
    ax.scatter(B[class_mask,0], B[class_mask,1], B[class_mask,2], label=c, color=color_array[i], alpha=.75)
    i += 1


ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend(loc='upper left')
ax.view_init(30, 120)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.tight_layout()
plt.savefig('Figures/3dprojection.pgf')
plt.show()

print(np.corrcoef(B[:,0], B[:,1]))
print(np.corrcoef(B[:,0], B[:,2]))
print(np.corrcoef(B[:,1], B[:,2]))