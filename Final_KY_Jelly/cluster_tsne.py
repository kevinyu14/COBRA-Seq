# Import packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import mode

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Modifiable settings:

# Load the data
data = np.loadtxt("results0.7threshold8444cells.txt.gz")

# Collect the names of genes, which are separate by an ";"
dimf = open("dimensions_of_results0.7threshold8444cells.txt", "r")
dimnames = dimf.readline().split(";")

# Make this so that cell nums are on the rows to cluster cells
data = data - np.min(data)
data = data + 1
for i in range(len(data)):
    data[i] = np.log10(data[i]) - np.log10(np.ones(len(data[i])) * mode(data[i])[0])
data = data.T

# kmeans = KMeans(n_clusters = 4)
# kmeans.fit(data)

# Subtract mean
centered = (data.T - np.mean(data, 1))

# Covariance matrix
cov = np.cov(centered)

# Get eigenvalues and eigenvectors
eigVal, eigVec = np.linalg.eig(cov)

# Sort eigenvalues and eigenvectors by eigenvalues
indxs = np.flip(np.argsort(eigVal))
eigVec = eigVec[:, indxs]
eigVal = eigVal[indxs]

reduced = np.matmul(centered.T, eigVec[:, 0:3])

clusters = 6

reduced_tsne = TSNE().fit_transform(reduced)

print(reduced_tsne.shape)

# plt.scatter(reduced_tsne[:, 0], reduced_tsne[:, 1])
# plt.show()

__, kIndx = kmeans2(data, clusters, minit = "points")

for i in range(clusters):
    clusterReduced = reduced_tsne[np.where(kIndx == i)]
    plt.scatter(clusterReduced[:, 0], clusterReduced[:, 1], s = 3)
plt.title("PCA of Gene Expression, Colored by K-Means MI with 6 Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()


