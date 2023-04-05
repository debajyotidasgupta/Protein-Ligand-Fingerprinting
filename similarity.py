import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from dotenv import dotenv_values

import deepchem as dc
from deepchem.utils import download_url, load_from_disk

from fingerprint import *

config = dotenv_values(".env")
data_dir = config['DATA_ROOT']
fingerprints_dir = config['FINGERPRINTS_DIR']
model_path = config['MODEL_PATH']
random_seed = int(config['RANDOM_SEED'])
output_dir = config['OUTPUT_DATA']
max_pdbs = int(config['MAX_PDBS'])

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
dataset = raw_dataset[['pdb_id']]

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

for _, row in dataset.iterrows():
    pdb_id = row['pdb_id']

    pdb_file = os.path.join(protein_ligand_dir, f"{pdb_id}.pdb")
    feature_output_file = os.path.join(fingerprints_dir, f"{pdb_id}.txt")

    if os.path.exists(feature_output_file):
        fingerprint_array = np.loadtxt(feature_output_file)
        pdb_ids.append(pdb_id)
        features.append(fingerprint_array)

    else:
        continue
        # Create a new instance of the class
        protein_ligand_complex = ProteinLigandSideChainComplex()

        # Load the PDB file
        protein_ligand_complex.load_pdb(pdb_file)

        # obtain the protein fingerprint
        fingerprint = NeighbourFingerprint(protein_ligand_complex)

        # obtain fixed length fingerprint (256 here)
        fingerprint_array = encode(
            fingerprint.get_fingerprint(), model_path=model_path)

        np.savetxt(feature_output_file, fingerprint_array, fmt='%1.8f')

    # pdb_ids.append(pdb_id)
    # features.append(fingerprint_array)

cosine_scores = []
euclidean_scores = []


def cosine_similarity(feature1, feature2):
    return np.dot(feature1, feature2)/(norm(feature1)*norm(feature2))


def euclidean_distance(feature1, feature2):
    return norm(feature1 - feature2)


for i in range(0, len(features)):

    if i == 0:
        print("Processing compound ", end='')

    if i % 10 == 0:
        print(i, end=' ')

    for j in range(i+1, len(features)):
        cosine_scores.append(cosine_similarity(features[i], features[j]))
        euclidean_scores.append(euclidean_distance(features[i], features[j]))

print("Done!")
print("# of scores : ", len(cosine_scores))

with open(os.path.join(output_dir, 'euclidean.txt'), 'w') as f:
    f.write(str(euclidean_scores))

with open(os.path.join(output_dir, 'cosine.txt'), 'w') as f:
    f.write(str(cosine_scores))

mybins = [x * 0.01 for x in range(101)]

# fig = plt.figure(figsize=(8,4), dpi=300)

# plt.subplot(1, 2, 1)
# plt.title("Distribution")
# plt.hist(cosine_scores, bins=mybins)

# plt.subplot(1, 2, 2)
# plt.title("Cumulative Distribution")
# plt.hist(cosine_scores, bins=mybins, density=True, cumulative=1)
# plt.plot([0,1],[0.95,0.95])

# plt.show()

for i in range(21):
    thresh = i / 20
    num_similar_pairs = len([x for x in cosine_scores if x >= thresh])
    prob = num_similar_pairs / len(cosine_scores) * 100
    print("%.3f %8d (%8.4f %%)" % (thresh, num_similar_pairs, round(prob, 4)))

print("Average:", sum(cosine_scores)/len(cosine_scores))
