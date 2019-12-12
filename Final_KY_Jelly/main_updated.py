import cobra
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import numpy as np
import multiprocessing as mp
from operator import itemgetter
import time

start_time = time.time()

# modifiable variables: cell # (max 8444), gene # (max 1800), threshold is % unique cells
cells = 8444
genes = 1800
threshold = .7
threads = mp.cpu_count() - 1
plot = False

# import model and make objective atp_c
# ALCD2X, etoh_c
modelOriginal = cobra.io.load_matlab_model('Recon3D.mat')
rxnName = 'atp_drain'
met_name = 'atp_c'
stoich = [-1]
tempRxn = cobra.Reaction(rxnName)
tempDrainMetDict = {modelOriginal.metabolites.get_by_id(met_name): s for name, s in zip(met_name, stoich)}
tempRxn.add_metabolites(tempDrainMetDict)
modelOriginal.add_reaction(tempRxn)
modelOriginal.objective = rxnName


def optimize_for_gene(name, expression):
    # return flux balance analysis result for a particular gene and cell #
    print('model start')
    global modelOriginal
    with modelOriginal as model:
        # get expression data from the scRNAseq data set
        # expression = data.loc[name][cell_num]
        # retrieve the reaction for the gene
        reactions = model.genes.get_by_id(gene_info[name]).reactions
        # change bounds for all reactions associated with the gene
        for j in reactions:
            j.lower_bound = 2**expression * 100
            j.upper_bound = 2**expression * 100
        fbas = model.optimize('maximize')
        # return gene name, cell #, and objective value so that we can recover
        # results from multiprocessing
        return [name, fbas.objective_value]


# read in scRNAseq data set
data = pd.read_csv('GSE115469_Data.csv', index_col=0)
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
# generate 50 random numbers to select 50 random cells to observe
# cell_nums = random.sample(range(len(list(data.loc[gene_matches[0]]))), 750)
# gene_nums = random.sample(range(len(gene_matches)), 10)
# multiprocessing w/ 3 threads
p = mp.Pool(threads)
# do FBA on the first 10 genes to make it faster for now
for num in range(len(gene_matches)):
    print('starting async')
    # find unique expression levels
    unique_cells, ucind = np.unique(data.loc[gene_matches[num]][:cells], return_inverse=True)
    if len(unique_cells) < threshold*cells:
        print('skipping gene %i' %(num))
        continue
    for i in range(len(unique_cells)):
        # helps to check which threads are running atm
        print("gene #: %d cell #: %d" % (num, i))
        print('starting async')
        # find the place where the unique expression levels are 
        cell_locs = [index for index in range(len(ucind)) if ucind[index] == i]
        # put the ApplyResult object in a list
        temp_result = p.apply_async(optimize_for_gene, args=(gene_matches[num], unique_cells[i]))
        # record instances of unique expression level results
        for ind in cell_locs:
            results.append([temp_result, num, ind])
# we're not using these threads anymore so we can move on
p.close()
# wait for all the threads to finish before we start converting results to
# a usable form
p.join()
# fetch the results of the ApplyResult object for the entire list
# make the ApplyResults into aa pandas dataframe to make it easier to sort
results_pd = pd.DataFrame.from_records(results)
# sort by gene # and cell # within gene #
results_pd = results_pd.sort_values([1, 2])
results_pd = results_pd.reset_index()
# get the result
results_fetched = [[results_pd.loc[i][0].get(), results_pd.loc[i][1], results_pd.loc[i][2]]
                   for i in range(results_pd.shape[0])]
# make it a pandas data frame so its easier to transform
df = pd.DataFrame.from_records(results_fetched)
# sort by the gene name first, then by the cell number within the gene name
df.sort_values(by=[1, 2])
# pivot converts a 1x10,0000 list to a 10x1,000 array with rownames that match
# unique values of column 0, and column names that match unique values of column 1
# and values that match column 2
df = df.pivot(index=1, columns=2, values=0)
# the dimnames should match the unique values of column 0 (gene names)
for i in range(int(results_pd.shape[0]/cells)):
    dimnames.append(df.iloc[i][0][0])
# convert the results back into a numpy array so that plotting is easer.
results_T = np.array(df.applymap(lambda x: x[1]))
filename = met_name + 'results' + str(threshold) + 'threshold' + str(cells) + 'cells' + '.txt.gz'
dimfilename = met_name + 'dimensions_of_results' + str(threshold) + 'threshold' + str(cells) + 'cells' + '.txt'
np.savetxt(filename, results_T)
dimf = open(dimfilename, 'w')
for i in dimnames:
    dimf.write(i + ';')
dimf.close()
print(time.time()-start_time)
if plot:
    print('plotting')
    # subtract the mode to deal with FBA values that are uninteresting
    for i in range(len(results_T)):
        results_T[i] -= np.ones(len(results_T[i])) * scipy.stats.mode(results_T[i])[0]
    # set up 100 plots to plot pairwise gene results
    fig, axs = plt.subplots(len(gene_matches[:genes]), len(gene_matches[:genes]))
    # plot all the results
    for i in range(len(results_T[:genes])):
        for j in range(len(results_T[:genes])):
            # print(results_T[i])
            axs[i, j].scatter(results_T[i], results_T[j])
            axs[i, j].set_xlabel(dimnames[j])
            axs[i, j].set_ylabel(dimnames[i])
    plt.show()
