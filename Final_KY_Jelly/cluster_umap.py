import scipy.cluster.vq as vq
import scipy.stats
import numpy as np
import umap
import matplotlib.pyplot as plt


# modifiable settings: cluster #, PC #
clusternum = 4

# load the data
data = np.loadtxt('atp_cresults0.7threshold2000cells.txt.gz')
# collect the names of genes
dimf = open('atp_cdimensions_of_results0.7threshold2000cells.txt', 'r')
dimnames = dimf.readline()
# gene names are separated by ;s
dimnames = dimnames.split(';')
# make this so that cell nums are on the rows to cluster cells
#data = data - np.min(data)
#data = data + 1
#for i in range(len(data)):
#    data[i] = np.log10(data[i]) - np.log10(np.ones(len(data[i])) * scipy.stats.mode(data[i])[0])
data = data.T
# cluster with kmeans to make 12 clusters
data = vq.whiten(data)
cen, l = vq.kmeans2(data, k=clusternum, minit='points', iter=200000)
# convert the cluster assignments to usable form
l = np.array(l)
l = l[:, np.newaxis]
l = np.transpose(l)

reducer = umap.UMAP()
im = reducer.fit_transform(data)
c = ['aqua', 'chartreuse', 'cyan', 'darkgreen', 'fuchsia', 'gold', 'indigo', 'ivory', 'orange', 'teal']
colors = [c[i] for i in l[0]]
plt.scatter(im[:, 0], im[:, 1], c=colors)
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()