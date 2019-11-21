import cobra
import matplotlib
import pandas as pd
import requests
import scipy.stats
import numpy as np
import multiprocessing as mp
p = mp.Pool(3)

def optimize_for_gene(name, cell_num, gene_num):
    with modelOriginal as model:
        expression = data.loc[name][cell_num]
        reactions = model.genes.get_by_id(gene_info[name]).reactions
        for j in reactions:
            j.lower_bound = expression * 100
            j.upper_bound = expression * 100
        FBASolution = model.optimize('maximize')
        results[cell_num, gene_num] = FBASolution.objective_value


modelOriginal = cobra.io.load_matlab_model('Recon3D.mat')
data = pd.read_csv('GSE115469_Data.csv')
genes_in_sc = data.index
with modelOriginal as model:
    rxns = model.genes
    gene_info = {}
    count = 0
    for i in rxns:
        url = 'http://bigg.ucsd.edu/api/v2/models/Recon3D/genes/' + str(i)
        r = requests.get(url)
        gene_info[r.json()['name']] = i.id
        count += 1
gene_matches = [genes_in_sc[i] for i in range(len(genes_in_sc)) if genes_in_sc[i] in gene_info.keys()]


num_cells = 1000
results = np.zeros((num_cells, len(gene_matches[:10])))
dimnames = []
for num in range(len(gene_matches[:10])):
    for i in range(len(list(data.loc[gene_matches[0]][:1000]))):
        print("gene #: %d cell #: %d" %(num, i))
        dimnames.append(gene_matches[num])
        temp_gene_name = gene_matches[num]
        optimize_for_gene(gene_matches(num), i)


results_T = results.T
for i in range(len(results_T)):
    results_T[i] -= np.ones(len(results_T[i])) * scipy.stats.mode(results_T[i])[0]
matplotlib.rcParams.update({"figure.figsize": (20, 15)})
fig, axs = plt.subplots(10, 10)
for i in range(len(results_T)):
    for j in range(len(results_T)):
        axs[i, j].scatter(results_T[i], results_T[j])
        axs[i, j].set_xlabel(dimnames[i * 1000])
        axs[i, j].set_ylabel(dimnames[j * 1000])
matplotlib.pyplot.tight_layout()
