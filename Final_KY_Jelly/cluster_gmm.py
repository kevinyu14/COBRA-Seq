import scipy.cluster.vq as vq
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture

# modifiable settings: cluster #, PC #
clusternum = 1

# load the data
data = np.loadtxt('atp_cresults0.9threshold2000cells.txt.gz')
# collect the names of genes
dimf = open('atp_cdimensions_of_results0.9threshold2000cells.txt', 'r')
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

from matplotlib.patches import Ellipse


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


gmm = GaussianMixture(n_components=8, random_state=42)
plot_gmm(gmm, pC_t)
