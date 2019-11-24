import cobra
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import numpy as np
import multiprocessing as mp

# import model
modelOriginal = cobra.io.load_matlab_model('Recon3D.mat')


def optimize_for_gene(name, cell_num):
    # return flux balance analysis result for a particular gene and cell #
    print('model start')
    global modelOriginal
    with modelOriginal as model:
        # get expression data from the scRNAseq data set
        expression = data.loc[name][cell_num]
        # retrieve the reaction for the gene
        reactions = model.genes.get_by_id(gene_info[name]).reactions
        # change bounds for all reactions associated with the gene
        for j in reactions:
            j.lower_bound = expression * 100
            j.upper_bound = expression * 100
        fbas = model.optimize('maximize')
        # return gene name, cell #, and objective value so that we can recover
        # results from multiprocessing
        return [name, cell_num, fbas.objective_value]


# read in scRNAseq data set
data = pd.read_csv('trunc_data.csv', index_col=0)
# gene names should be the 0th column, which is the index column
genes_in_sc = data.index
# read in the map from gene name -> model name
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
# find all genes that exist in the scRNAseq set and in the model
gene_matches = [genes_in_sc[i] for i in range(len(genes_in_sc)) if genes_in_sc[i] in gene_info.keys()]
# prepare a list to receive the ApplyResult objects from multiprocessing
results = []
# prepare a list to make plotting axes easier
dimnames = []
print('starting models')
# multiprocessing w/ 3 threads
p = mp.Pool(3)
# do FBA on the first 10 genes to make it faster for now
for num in range(len(gene_matches[:10])):
    print('starting async')
    # do it on the first 1000 genes that match so its faster
    for i in range(len(list(data.loc[gene_matches[0]][:1000]))):
        # helps to check which threads are running atm
        print("gene #: %d cell #: %d" % (num, i))
        print('starting async')
        # put the ApplyResult object in a list
        results.append(p.apply_async(optimize_for_gene, args=(gene_matches[num], i)))
# we're not using these threads anymore so we can move on
p.close()
# wait for all the threads to finish before we start converting results to
# a usable form
p.join()
# fetch the results of the ApplyResult object for the entire list
results_fetched = [i.get() for i in results]
# make it a pandas data frame so its easier to transform
df = pd.DataFrame.from_records(results_fetched)
# sort by the gene name first, then by the cell number within the gene name
df.sort_values(by=[0, 1])
# pivot converts a 1x10,0000 list to a 10x1,000 array with rownames that match
# unique values of column 0, and column names that match unique values of column 1
# and values that match column 2
df = df.pivot(index=0, columns=1, values=2)
# the dimnames should match the unique values of column 0 (gene names)
dimnames = df.index.values
# convert the results back into a numpy array so that plotting is easer.
results_T = np.array(df.values.tolist())

print('plotting')
# subtract the mode to deal with FBA values that are uninteresting
for i in range(len(results_T)):
    results_T[i] -= np.ones(len(results_T[i])) * scipy.stats.mode(results_T[i])[0]
# set up 100 plots to plot pairwise gene results
fig, axs = plt.subplots(10, 10)
# plot all the results
for i in range(len(results_T)):
    for j in range(len(results_T)):
        #print(results_T[i])
        axs[i, j].scatter(results_T[i], results_T[j])
        axs[i, j].set_xlabel(dimnames[j])
        axs[i, j].set_ylabel(dimnames[i])
plt.show()
