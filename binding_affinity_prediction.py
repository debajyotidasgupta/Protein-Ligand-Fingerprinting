import os
import numpy as np
import requests
from dotenv import dotenv_values
from sklearn.ensemble import RandomForestRegressor
import deepchem as dc
from deepchem.utils import download_url, load_from_disk
from deepchem.utils.evaluate import Evaluator

from fingerprint import *

config = dotenv_values(".env")
data_dir = config['DATA_ROOT']
fingerprints_dir = config['FINGERPRINTS_DIR']
intermediate_dir = config['INTERMEDIATE_DIR']
model_path = config['MODEL_PATH']
random_seed = int(config['RANDOM_SEED'])
max_pdbs = int(config['MAX_PDBS'])
output_dir = config['OUTPUT_DATA']

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
if not os.path.isdir(fingerprints_dir):
    os.mkdir(fingerprints_dir)

dataset_file = os.path.join(data_dir, "pdbbind_core_df.csv.gz")

if not os.path.exists(dataset_file):
    print("File does not exist. Downloading file...")
    download_url(
        "https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/pdbbind_core_df.csv.gz", dest_dir=data_dir)
    print("File downloaded!")

raw_dataset = load_from_disk(dataset_file)
dataset = raw_dataset[['pdb_id', 'label']]

protein_ligand_dir = os.path.join(data_dir, 'pdb_files')
if not os.path.isdir(protein_ligand_dir):
    os.mkdir(protein_ligand_dir)

pdb_ids = dataset['pdb_id']

count = 0

for pdb_id in pdb_ids:
    if count >= max_pdbs:
        break
    count += 1
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

count = 0

for _, row in dataset.iterrows():
    if count >= max_pdbs:
        break
    count += 1
    pdb_id = row['pdb_id']
    label = row['label']

    pdb_file = os.path.join(protein_ligand_dir, f"{pdb_id}.pdb")
    feature_output_file = os.path.join(fingerprints_dir, f"{pdb_id}.txt")
    neighbour_feature_output_file = os.path.join(
        intermediate_dir, f"{pdb_id}.txt")

    if os.path.exists(feature_output_file):
        fingerprint_array = np.loadtxt(feature_output_file)

    elif os.path.exists(neighbour_feature_output_file):
        fingerprint_array = np.loadtxt(neighbour_feature_output_file)

        # obtain fixed length fingerprint (256 here)
        fingerprint_array = encode(fingerprint_array, model_path=model_path)

        np.savetxt(feature_output_file, fingerprint_array, fmt='%1.8f')

    else:
        # Create a new instance of the class
        protein_ligand_complex = ProteinLigandSideChainComplex()

        # Load the PDB file
        protein_ligand_complex.load_pdb(pdb_file)

        # obtain the protein fingerprint
        fingerprint = NeighbourFingerprint(protein_ligand_complex)

        # save meighbout fingerprint in intermediate folder
        np.savetxt(neighbour_feature_output_file,
                   fingerprint.get_fingerprint(), fmt='%1.8f')

        # obtain fixed length fingerprint (256 here)
        fingerprint_array = encode(
            fingerprint.get_fingerprint(), model_path=model_path)

        # save the final fingerprint in fingerprints folder
        np.savetxt(feature_output_file, fingerprint_array, fmt='%1.8f')

    pdb_ids.append(pdb_id)
    features.append(fingerprint_array)
    labels.append(label)

pdb_ids = np.array(pdb_ids)

# pad to maximum length if all fingerprints are not of same length
# max_shape = tuple(max(i) for i in zip(*[a.shape for a in features]))
# padded_arrays = [np.pad(a, [(0, max_shape[0] - a.shape[0]), (0, max_shape[1] - a.shape[1])], 'constant') for a in features]
# padded_arrays = [a.reshape(a.shape[0]*a.shape[1]) for a in padded_arrays]
features = np.stack(features)
labels = np.array(labels)

# create dataset for prediction of binding affinity
dataset = dc.data.NumpyDataset(X=features, y=labels, ids=pdb_ids)
train_dataset, test_dataset = dc.splits.RandomSplitter(
).train_test_split(dataset, seed=random_seed)

# define the model
sk_model = RandomForestRegressor(n_estimators=35, max_features="sqrt")
sk_model.random_state = random_seed
model = dc.models.SklearnModel(sk_model)
# train
model.fit(train_dataset)

# use R^2 score as metric for comparision
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

train_comparision = list(zip(model.predict(train_dataset), train_dataset.y))
test_comparision = list(zip(model.predict(test_dataset), test_dataset.y))

open(os.path.join(output_dir, 'binding_train_comparision.txt'),
     'w').write(str(train_comparision))
open(os.path.join(output_dir, 'binding_test_comparision.txt'),
     'w').write(str(test_comparision))

evaluator = Evaluator(model, train_dataset, [])
train_r2score = evaluator.compute_model_performance([metric])
print(f'RF Train set R^2 {train_r2score["pearson_r2_score"]}')

evaluator = Evaluator(model, test_dataset, [])
test_r2score = evaluator.compute_model_performance([metric])
print(f'RF Test set R^2 {test_r2score["pearson_r2_score"]}')
