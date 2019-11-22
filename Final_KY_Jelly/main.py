import cobra
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import numpy as np
import multiprocessing as mp
import functools


gene_results = []


def optimize_for_gene(name, cell_num):
    with modelOriginal as model:
        expression = data.loc[name][cell_num]
        reactions = model.genes.get_by_id(gene_info[name]).reactions
        for j in reactions:
            j.lower_bound = expression * 100
            j.upper_bound = expression * 100
        fbas = model.optimize('maximize')
        return [name, fbas.objective_value]


def collect_results(result):
    global gene_results
    gene_results.append(result)


data = pd.read_csv('GSE115469_Data.csv', index_col = 0)
genes_in_sc = data.index
f = open('map.txt', 'r')
dict_temp = f.readlines()
list_dict = dict_temp[0].split(';')
gene_info = {}
for i in list_dict:
    if i == '':
        continue
    tempkey = i.split(',')[0]
    tempval = i.split(',')[1]
    gene_info[tempkey] = tempval
print('loading model')
modelOriginal = cobra.io.load_matlab_model('Recon3D.mat')
print(genes_in_sc)
print(gene_info.keys())

gene_matches = [genes_in_sc[i] for i in range(len(genes_in_sc)) if genes_in_sc[i] in gene_info.keys()]

print(gene_matches)
num_cells = 1000
results = np.zeros((num_cells, len(gene_matches[:10])))
dimnames = []
print('starting models')
p = mp.Pool(3)
for num in range(len(gene_matches[:10])):
    print('starting async')
    for i in range(len(list(data.loc[gene_matches[0]][:1000]))):
        print("gene #: %d cell #: %d" % (num, i))
        dimnames.append(gene_matches[num])
        temp_gene_name = gene_matches[num]
        print('starting async')
        p.apply_async(optimize_for_gene, args=(gene_matches(num), i), callback=collect_results)

    #results = [p.apply(optimize_for_gene, args=(gene_matches(num), i)) for i in range(num_cells)]
    #p.join()
    #temp = results.copy()
    #gene_results.append(temp)
    #results = []
p.join()
p.close()


print('plotting')
results_T = results.T
for i in range(len(results_T)):
    results_T[i] -= np.ones(len(results_T[i])) * scipy.stats.mode(results_T[i])[0]
fig, axs = plt.subplots(10, 10)
for i in range(len(results_T)):
    for j in range(len(results_T)):
        print(results_T[i])
        axs[i, j].scatter(results_T[i], results_T[j])
        axs[i, j].set_xlabel(dimnames[i * 1000])
        axs[i, j].set_ylabel(dimnames[j * 1000])
plt.show()
