import scipy.cluster.hierarchy as hier
import scipy.stats
import numpy as np
import umap
import matplotlib.pyplot as plt


# modifiable settings: cluster #, PC #
clusternum = 6

# load the data
data = np.loadtxt('atp_cresults0.7threshold2000cells.txt.gz')
# collect the names of genes
dimf = open('atp_cdimensions_of_results0.7threshold2000cells.txt', 'r')
dimnames = dimf.readline()
# gene names are separated by ;s
dimnames = dimnames.split(';')[:-1]
print(dimnames)
# make this so that cell nums are on the rows to cluster cells
data = data - np.min(data)
data = data + 1
for i in range(len(data)):
    data[i] = np.log10(data[i]) - np.log10(np.ones(len(data[i])) * scipy.stats.mode(data[i])[0])
link = hier.linkage(data, method = 'ward')
cut = hier.cut_tree(link)
R = hier.dendrogram(link, p = 3, orientation = 'left', truncate_mode='lastp',
                leaf_font_size = 15, get_leaves = True, labels=dimnames)
plt.show()
