import cobra
import matplotlib
import pandas as pd
import requests
import scipy.stats
import numpy as np

#test

modelOriginal = cobra.io.load_matlab_model('Recon3D.mat')
data = pd.read_csv('GSE115469_Data.csv')
genes_in_sc = data.index
with modelOriginal as model:
	rxns = model.genes
	gene_info = {}
	count = 0
	for i in rxns:
		url = 'http://bigg.ucsd.edu/api/v2/models/Recon3D/genes'+str(i)
		r = requests.get(url)
		gene_info[r.json()['name']] = i.id
		count += 1
gene_matches = [genes_in_sc[i] for i in range(len(genes_in_sc)) if genes_in_sc[i] in gene_info.keys()]
num_cells = 1000
results = np.zeros((num_cells), len(gene_matches[:10]))
dimnames = []
for num in range(len(gene_matches[:10])):
	for i in range(len(list(data.loc[gene_matches[0]][:1000]))):
		with modelOriginal as model:
			dimnames.append(gene_matches[num])
			exp = data.loc[gene_matches[num]][i]
			reactions = model.genes.get_by_id(gene_info[gene_matches[num]]).reactions
			for j in reactions:
				j.upper_bound = exp*100
				j.lower_bound = exp*100
			FBASolution = model.optimize('maximize')
			results[i,num] = FBASolution.objective_value

results_T = results.T
for i in range(len(results_T)):
	results_T[i] -= np.ones(len(results_T[i]))*scipy.stats.mode(results_T[i])[0]
matplotlib.rcParams.update({"figure.figsize":(20,15)})
fig, axs = plt.subplots(10,10)
for i in range(len(results_T)):
	for j in range(len(results_T)):
		axs[i, j].scatter(results_T[i], results_T[j])
		axs[i, j].set_xlabel(dimnames[i*1000])
		axs[i, j].set_ylabel(dimnames[j*1000])
matplotlib.pyplot.tight_layout()
