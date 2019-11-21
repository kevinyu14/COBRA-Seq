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
f = open('map.txt', 'w')
f.write(str(gene_info))
f.close()


