import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from dotenv import dotenv_values
from deepchem.utils import download_url, load_from_disk
import requests

from fingerprint import *
from fingerprint import *

data_dir = './data'
fingerprints_dir = './output/fingerprints'
models_dir = './models/AutoencoderTransformer_4.pt'

def generate_neighbour_fingerprints():
    dataset_file = os.path.join(data_dir, "pdbbind_core_df.csv.gz")

    if not os.path.exists(dataset_file):
        print("File does not exist. Downloading file...")
        download_url("https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/pdbbind_core_df.csv.gz", dest_dir=data_dir)
        print("File downloaded!")
    
    raw_dataset = load_from_disk(dataset_file)
    dataset = raw_dataset[['pdb_id']]

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

    for pbd_id in pdb_ids:

        pdb_file = os.path.join(protein_ligand_dir, f"{pdb_id}.pdb")
        feature_output_file = os.path.join(fingerprints_dir, f"{pdb_id}.txt")

        if os.path.exists(feature_output_file):
            continue
        
        else:
            # Create a new instance of the class
            protein_ligand_complex = ProteinLigandSideChainComplex()

            # Load the PDB file
            protein_ligand_complex.load_pdb(pdb_file)

            # obtain the protein fingerprint
            fingerprint = NeighbourFingerprint(protein_ligand_complex)
            fingerprint_array = fingerprint.get_fingerprint()

            np.savetxt(feature_output_file, fingerprint_array, fmt='%1.8f')

def train(input_dim=5, hidden_dim=32, num_layers=2, num_heads=5, output_dim=256, epochs=5):
    model = AutoencoderTransformer(input_dim=input_dim, hidden_dim=hidden_dim, 
                                   num_layers=num_layers, num_heads=num_heads, output_dim=output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    fingerprints = []
    for file in os.listdir(fingerprints_dir):
        fingerprints.append(np.loadtxt(os.path.join(fingerprints_dir, file)))

    train_loader = [torch.from_numpy(x) for x in fingerprints]
    train_loader = [torch.unsqueeze(x, 0) for x in train_loader]
    train_loader = [x.to(torch.float) for x in train_loader]

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs = data
            optimizer.zero_grad()
            _, decoded = model(inputs)
            loss = criterion(inputs, decoded)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20== 19:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

        torch.save(model, os.path.join(models_dir,f"AutoencoderTransformer_{epoch}.pt"))

if __name__ == "__main__":
    config = dotenv_values(".env")
    data_dir = config['DATA_ROOT']
    fingerprints_dir = config['FINGERPRINTS_DIR']
    models_dir = config['MODEL_PATH']

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    if not os.path.isdir(fingerprints_dir):
        os.mkdir(fingerprints_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dim', default=5)
    parser.add_argument('-hi', '--hidden_dim', default=32)
    parser.add_argument('-l', '--num_layers', default=2)
    parser.add_argument('-nh', '--num_heads', default=5)
    parser.add_argument('-o', '--output_dim', default=256)
    parser.add_argument('-e', '--epochs', default=5)
    args = parser.parse_args()

    train(args.input_dim, args.hidden_dim, args.num_layers, args.num_heads, args.output_dim, args.epochs)
