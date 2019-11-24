import cobra
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import numpy as np
import multiprocessing as mp


modelOriginal = cobra.io.load_matlab_model('Recon3D.mat')


def optimize_for_gene(name, cell_num):
    print('model start')
    global modelOriginal
    with modelOriginal as model:
        expression = data.loc[name][cell_num]
        reactions = model.genes.get_by_id(gene_info[name]).reactions
        for j in reactions:
            j.lower_bound = expression * 100
            j.upper_bound = expression * 100
        fbas = model.optimize('maximize')
        return [name, cell_num, fbas.objective_value]


data = pd.read_csv('trunc_data.csv', index_col=0)
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
f.close()
print('loading model')

print(genes_in_sc)
print(gene_info.keys())

gene_matches = [genes_in_sc[i] for i in range(len(genes_in_sc)) if genes_in_sc[i] in gene_info.keys()]

print(gene_matches)
num_cells = 1000
results = []
dimnames = []
print('starting models')
p = mp.Pool(3)
for num in range(len(gene_matches[:10])):
    print('starting async')
    for i in range(len(list(data.loc[gene_matches[0]][:1000]))):
        print("gene #: %d cell #: %d" % (num, i))
        print('starting async')
        results.append(p.apply_async(optimize_for_gene, args=(gene_matches[num], i)))
p.close()
p.join()
results_fetched = [i.get() for i in results]
df = pd.DataFrame.from_records(results_fetched)
df.sort_values(by=[0, 1])  # sort by the gene name first, then by the cell number within the gene name
df = df.pivot(index=0, columns=1, values=2)
dimnames = df.index.values
results_T = df.values.to_list()


print('plotting')
for i in range(len(results_T)):
    results_T[i] -= np.ones(len(results_T[i])) * scipy.stats.mode(results_T[i])[0]
fig, axs = plt.subplots(10, 10)
for i in range(len(results_T)):
    for j in range(len(results_T)):
        #print(results_T[i])
        axs[i, j].scatter(results_T[i], results_T[j])
        axs[i, j].set_xlabel(dimnames[i])
        axs[i, j].set_ylabel(dimnames[j])
plt.show()
