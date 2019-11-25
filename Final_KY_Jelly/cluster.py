import scipy.cluster.vq as vq
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


# load the data
data = np.loadtxt('results.txt')
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
cen, l = vq.kmeans2(data, k=6, minit='points')
# convert the cluster assignments to usable form
l = np.array(l)
l = l[:, np.newaxis]
# sort the data by cluster assignment
tdata= np.concatenate((l, data), 1)
tdata = tdata[tdata[:, 0].argsort()]
hlinepos = []
for i in range(6):
    tempin = np.where(tdata[:, 0] == i)
    if len(tempin) != 0:
        hlinepos.append(tempin[0][0])
        # print(tempin[0][0])
# remove the cluster assignments from the data
k_means_sorted = tdata[:, 1:]
# plot it
#plt.imshow(k_means_sorted, cmap='bwr', aspect='auto', vmin=-.01, vmax=.01)
#cbar = plt.colorbar()
# make the gene names the x axis labels
#plt.xticks(range(0, len(dimnames[:-1])), dimnames[:-1], rotation=45)
print(len(hlinepos))
#for i in range(6):
    # print(hlinepos[i])
#    plt.axhline(hlinepos[i], color="black")

submean = data - np.mean(data, 1).reshape((data.shape[0], 1))
submean = submean.T
covmat = np.cov(submean)
W, V = np.linalg.eig(covmat)
idx = np.flip(np.argsort(W))
V = V[:, idx]
W = W[idx]
evr = W/np.sum(W)
#plt.bar(range(10), evr)
R = np.matmul(submean.T, V[:, 0:2]).T
cluster = list(l)
R0 = np.matrix(R[0,:])
R1 = np.matrix(R[1,:])
pC_t = np.vstack((R0, R1)).T
print(pC_t)
targets = [0, 1, 2, 3, 4, 5]
colors = ['r', 'g', 'b', 'y', 'c', 'm']
count = 1
for target, color in zip(targets,colors):
    indices = [i for i in range(len(cluster)) if cluster[i] == target]
    print(pC_t[2])
    print(indices)
    x = [pC_t[i,0] for i in indices]
    y = [pC_t[i,1] for i in indices]
    print(x, y)
    plt.scatter(x, y, c=color, s=50)
    count += 1
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA via Eigen')
plt.show()