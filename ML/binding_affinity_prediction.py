import os
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
import deepchem as dc
from deepchem.utils import download_url, load_from_disk
from deepchem.utils.evaluate import Evaluator

from fingerprint import *


random_seed = 42

data_dir = '../data'
output_dir = '../output'

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

dataset_file = os.path.join(data_dir, "pdbbind_core_df.csv.gz")

if not os.path.exists(dataset_file):
    print("File does not exist. Downloading file...")
    download_url("https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/pdbbind_core_df.csv.gz", dest_dir=data_dir)
    print("File downloaded!")

raw_dataset = load_from_disk(dataset_file)
dataset = raw_dataset[['pdb_id', 'label']]

protein_ligand_dir = os.path.join(data_dir, 'pdb_files')
if not os.path.isdir(protein_ligand_dir):
    os.mkdir(protein_ligand_dir)

pdb_ids = dataset['pdb_id']

for pdb_id in pdb_ids:
    pdb_file = os.path.join(protein_ligand_dir, f"{pdb_id}.pdb")
    if os.path.exists(pdb_file):
        continue
    url = f"http://files.rcsb.org/download/{pdb_id}.pdb"
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200:
        open(pdb_file, "wb").write(res.content)

pdb_ids = []
features = []
labels = []

i = 0
for _, row in dataset.iterrows():
    if i>=10:
        break
    i+=1
    pdb_id = row['pdb_id']
    label = row['label']

    pdb_file = os.path.join(protein_ligand_dir, f"{pdb_id}.pdb")

    # Create a new instance of the class
    protein_ligand_complex = ProteinLigandSideChainComplex()

    # Load the PDB file
    protein_ligand_complex.load_pdb(pdb_file)

    # obtain the protein fingerprint
    fingerprint = NeighbourFingerprint(protein_ligand_complex).get_fingerprint()

    pdb_ids.append(pdb_id)
    features.append(fingerprint)
    labels.append(label)

pdb_ids = np.array(pdb_ids)
# features = np.stack(features)
labels = np.array(labels)

dataset = dc.data.NumpyDataset(X=features, y=labels, ids=pdb_ids)
train_dataset, test_dataset = dc.splits.RandomSplitter().train_test_split(dataset, seed=random_seed)

sk_model = RandomForestRegressor(n_estimators=100, max_features="sqrt")
sk_model.random_state = random_seed
model = dc.models.SklearnModel(sk_model)
model.fit(train_dataset)

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

train_comparision = list(zip(model.predict(train_dataset), train_dataset.y))
test_comparision = list(zip(model.predict(test_dataset), test_dataset.y))

open(os.path.join(output_dir, 'train_comparision'), 'w').write(train_comparision)
open(os.path.join(output_dir, 'test_comparision'), 'w').write(test_comparision)

evaluator = Evaluator(model, train_dataset, [])
train_r2score = evaluator.compute_model_performance([metric])
print(f'RF Train set R^2 {train_r2score["pearson_r2_score"]}')

evaluator = Evaluator(model, test_dataset, [])
test_r2score = evaluator.compute_model_performance([metric])
print(f'RF Test set R^2 {test_r2score["pearson_r2_score"]}')
