import scipy.cluster.vq as vq
import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture

# modifiable settings: cluster #, PC #
clusternum = 1

# load the data
data = np.loadtxt('etoh_results0.7threshold2000cells.txt.gz')
# collect the names of genes
dimf = open('etoh_dimensions_of_results0.7threshold2000cells.txt', 'r')
dimnames = dimf.readlines()
# gene names are separated by ;s
dimnames = dimnames[0].split(';')
# make this so that cell nums are on the rows to cluster cells
data = data - np.min(data)
data = data + 1
for i in range(len(data)):
    data[i] = np.log10(data[i]) - np.log10(np.ones(len(data[i])) * scipy.stats.mode(data[i])[0])
data = data.T

# convert the cluster assignments to usable form
submean = data - np.mean(data, 1).reshape((data.shape[0], 1))
submean = submean.T
covmat = np.cov(submean)
W, V = np.linalg.eig(covmat)
idx = np.flip(np.argsort(W))
V = V[:, idx]
W = W[idx]
evr = W / np.sum(W)

R = np.matmul(submean.T, V[:, 0:2]).T
R0 = np.matrix(R[0, :])
R1 = np.matrix(R[1, :])
pC_t = np.vstack((R0, R1)).T
print(pC_t)

# cluster with gmm to make 12 clusters
gmm = GaussianMixture(n_components=8);
gmm.fit(pC_t)
# convert the cluster assignments to usable form
print(gmm.converged_)

# make plots

# display predicted scores by the model as a contour plot
x = np.linspace(-0.1, 1)
y = np.linspace(-0.5, 0.5)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
pct0 = np.array([])
pct1 = np.array([])
for i in range(len(pC_t)):
    pct0 = np.append(pct0, pC_t[i,0])
    pct1 = np.append(pct1, pC_t[i,1])

# plt.yscale('symlog')
# plt.xscale('symlog')


plt.scatter(pct0, pct1, .8)
plt.xlim(-0.1,0.4)
plt.ylim(-0.1,0.3)
plt.title('Negative log-likelihood predicted by a GMM')
# plt.axis('tight')
plt.show()
