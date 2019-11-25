import scipy
import numpy as np
import matplotlib.pyplot as plt


# load the data
data = np.loadtxt('results.txt.gz')
# collect the names of genes
dimf = open('dimensions_of_results.txt', 'r')
dimnames = dimf.readlines()
# gene names are separated by ;s
dimnames = dimnames.split(';')
# make this so that cell nums are on the rows to cluster cells
data = data.T
# take an initial look at the data
plt.imshow(data, cmap='bwr', aspect='auto')
plt.show()
w = scipy.cluster.vq.whiten(data)
cen, l = scipy.cluster.vq.kmeans2(w, minit='points')
l = np.array(l)
l = l[:,np.newaxis]
tdata= np.concatenate((l, data), 1)
tdata = tdata[tdata[:, 0].argsort()]
hlinepos =
