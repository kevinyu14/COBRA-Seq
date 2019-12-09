import scipy.cluster.vq as vq
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm


# modifiable settings: cluster #, PC #
clusternum = 6

# load the data
data = np.loadtxt('results8genes2000cells.txt.gz')
# collect the names of genes
dimf = open('dimensions_of_results8genes2000cells.txt', 'r')
dimnames = dimf.readlines()
# gene names are separated by ;s
dimnames = dimnames[0].split(';')
# make this so that cell nums are on the rows to cluster cells
data = data - np.min(data)
data = data + 1
for i in range(len(data)):
    data[i] = np.log10(data[i]) - np.log10(np.ones(len(data[i])) * scipy.stats.mode(data[i])[0])
data = data.T
# cluster with kmeans to make 12 clusters
cen, l = vq.kmeans2(data, k=clusternum, minit='points')
# convert the cluster assignments to usable form
l = np.array(l)
l = l[:, np.newaxis]
# sort the data by cluster assignment
tdata= np.concatenate((l, data), 1)
tdata = tdata[tdata[:, 0].argsort()]
hlinepos = []
for i in range(clusternum):
    tempin = np.where(tdata[:, 0] == i)
    if len(tempin) != 0:
        hlinepos.append(tempin[0][0])
# remove the cluster assignments from the data
k_means_sorted = tdata[:, 1:]
# make plots
fig, axs = plt.subplots(1, 3)
# plot it
im = axs[2].imshow(k_means_sorted, cmap='bwr', aspect='auto', vmin=k_means_sorted.min(), vmax=-k_means_sorted.min())
divider = make_axes_locatable(axs[2])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Log Fold-Change')
# make the gene names the x axis labels
axs[2].set_xticks(range(0, len(dimnames[:-1])))
axs[2].set_xticklabels(dimnames[:-1])
plt.setp(axs[2].xaxis.get_majorticklabels(), rotation=80, fontsize=5)
axs[2].set_ylabel('Cells')
axs[2].set_title('Histogram')
for i in range(clusternum):
    print(hlinepos[i])
    axs[2].axhline(hlinepos[i], color="black")

submean = data - np.mean(data, 1).reshape((data.shape[0], 1))
submean = submean.T
covmat = np.cov(submean)
W, V = np.linalg.eig(covmat)
idx = np.flip(np.argsort(W))
V = V[:, idx]
W = W[idx]
evr = W/np.sum(W)
axs[1].bar(range(len(evr)), evr)
axs[1].set_xlabel('Principal Component #')
axs[1].set_ylabel('Variance')
axs[1].set_title('Explained Variance')
R = np.matmul(submean.T, V[:, 0:2]).T
cluster = list(l)
R0 = np.matrix(R[0,:])
R1 = np.matrix(R[1,:])
pC_t = np.vstack((R0, R1)).T
print(pC_t)
targets = range(clusternum)
count = 1
for target in targets:
    color = "C"+str(targets[target])
    indices = [i for i in range(len(cluster)) if cluster[i] == target]
    x = [pC_t[i, 0] for i in indices]
    y = [pC_t[i, 1] for i in indices]
    axs[0].scatter(x, y, c=color, s=50)
    count += 1
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')
axs[0].set_title('PCA via Eigen')
plt.show()
fig.tight_layout()
