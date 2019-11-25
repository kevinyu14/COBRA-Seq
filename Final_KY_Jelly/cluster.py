import scipy.cluster.vq as vq
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


# load the data
data = np.loadtxt('results.txt.gz')
# collect the names of genes
dimf = open('dimensions_of_results.txt', 'r')
dimnames = dimf.readlines()
# gene names are separated by ;s
dimnames = dimnames[0].split(';')
# make this so that cell nums are on the rows to cluster cells
for i in range(len(data)):
    data[i] = np.log(data[i]) - np.log(np.ones(len(data[i])) * scipy.stats.mode(data[i])[0])
data = data.T
# cluster with kmeans to make 12 clusters
cen, l = vq.kmeans2(data, k=12, minit='points')
# convert the cluster assignments to usable form
l = np.array(l)
l = l[:, np.newaxis]
# sort the data by cluster assignment
tdata= np.concatenate((l, data), 1)
tdata = tdata[tdata[:, 0].argsort()]
# remove the cluster assignments from the data
k_means_sorted = tdata[:, 1:]
# plot it
plt.imshow(k_means_sorted, cmap='bwr', aspect='auto', vmin=-.01, vmax=.01)
cbar = plt.colorbar()
# make the gene names the x axis labels
plt.xticks(range(0, len(dimnames[:-1])), dimnames[:-1], rotation=45)
plt.show()
