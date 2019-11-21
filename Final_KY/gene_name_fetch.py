import cobra
import pandas as pd
import requests

modelOriginal = cobra.io.load_matlab_model('Recon3D.mat')
print('model')
data = pd.read_csv('GSE115469_Data.csv')
print('data')
genes_in_sc = data.index
print('starting json')
with modelOriginal as model:
    rxns = model.genes
    gene_info = {}
    count = 0
    for i in rxns:
        print(i)
        url = 'http://bigg.ucsd.edu/api/v2/models/Recon3D/genes/' + str(i)
        r = requests.get(url)
        gene_info[r.json()['name']] = i.id
        count += 1
f = open('map.txt', 'w')
for key,val in gene_info.items():
    f.write("%s,%s;" % (key, val))
f.close()


