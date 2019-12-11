import scipy.cluster.vq as vq
import scipy.stats
import numpy as np
import umap
import matplotlib.pyplot as plt


# modifiable settings: cluster #, PC #
clusternum = 8

# load the data
data = np.loadtxt('results0.7threshold8444cells.txt.gz')
# collect the names of genes
dimf = open('dimensions_of_results0.7threshold8444cells.txt', 'r')
dimnames = dimf.readline()
# gene names are separated by ;s
dimnames = dimnames.split(';')
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
l = np.transpose(l)
filename = 'clusters0.7threshold8444cells.txt.gz'
np.savetxt(filename, l)

reducer = umap.UMAP()
im = reducer.fit_transform(data)
c = ['aqua', 'black', 'chartreuse', 'cyan', 'darkgreen', 'fuchsia', 'gold', 'grey', 'indigo', 'ivory', 'orange', 'teal']
colors = [c[i] for i in l[0]]
plt.scatter(im[:, 0], im[:, 1], c=colors)
plt.show()