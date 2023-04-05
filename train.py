import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from dotenv import dotenv_values
from deepchem.utils import download_url, load_from_disk
import requests

from fingerprint import *
from fingerprint import *

data_dir = './data'
intermediate_dir = './output/intermediate'
fingerprints_dir = './output/fingerprints'
models_dir = './models'
max_pdbs = 100
random_seed = 42

def generate_neighbour_fingerprints():
    """
    generate the intermediate fingerprints for max_pdbs pdb ids
    based on neighbourhood fingerprinting
    """
    dataset_file = os.path.join(data_dir, "pdbbind_core_df.csv.gz")

    # download the pdbbind dataset
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

    # download the pdb files for pdb ids present in pdbbind dataset if not already present 
    for pdb_id in pdb_ids:
        pdb_file = os.path.join(protein_ligand_dir, f"{pdb_id}.pdb")
        if os.path.exists(pdb_file):
            continue
        url = f"http://files.rcsb.org/download/{pdb_id}.pdb"
        res = requests.get(url, allow_redirects=True)
        if res.status_code == 200:
            open(pdb_file, "wb").write(res.content)

    count = 0
    for pdb_id in pdb_ids:

        if count >= max_pdbs:
            break

        count+=1

        pdb_file = os.path.join(protein_ligand_dir, f"{pdb_id}.pdb")
        feature_output_file = os.path.join(intermediate_dir, f"{pdb_id}.txt")

        # if the intermediate neighbourhood fingerprint already exists continue
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

            # save the intermediate neighbourhood fiingerprint
            np.savetxt(feature_output_file, fingerprint_array, fmt='%1.8f')

def train(input_dim=5, hidden_dim=32, num_layers=2, num_heads=5, output_dim=256, epochs=10):
    """
    main code to train the AutoencoderTransformer
    """
    model = AutoencoderTransformer(input_dim=input_dim, hidden_dim=hidden_dim, 
                                   num_layers=num_layers, num_heads=num_heads, output_dim=output_dim)
    # train using mean squared error loss
    criterion = nn.MSELoss()
    # use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    fingerprints = []
    # load the intermediate neighbourhood fingerprints
    for file in os.listdir(intermediate_dir):
        fingerprints.append(np.loadtxt(os.path.join(intermediate_dir, file)))
    
    # perform a train test split in 80:20 ratio
    fingerprints_train, fingerprints_test, _, _ = train_test_split(fingerprints, [0 for x in fingerprints], test_size=0.2, random_state=random_seed)

    train_loader = [torch.from_numpy(x) for x in fingerprints_train]
    train_loader = [torch.unsqueeze(x, 0) for x in train_loader]
    train_loader = [x.to(torch.float) for x in train_loader]

    test_loader = [torch.from_numpy(x) for x in fingerprints_test]
    test_loader = [torch.unsqueeze(x, 0) for x in test_loader]
    test_loader = [x.to(torch.float) for x in test_loader]

    min_loss = 1000000.0

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        # TRAIN
        for i, data in enumerate(train_loader, 0):
            inputs = data
            optimizer.zero_grad()
            _, decoded = model(inputs)
            loss = criterion(inputs, decoded)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20== 19:
                print('[%d, %5d] training loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        
        model.eval()
        # EVALUATE
        for i, data in enumerate(test_loader, 0):
            inputs = data
            _, decoded = model(inputs)
            loss = criterion(inputs, decoded)
            if loss < min_loss:
                min_loss = loss
                print(f"min val loss = {min_loss}")
                # save the best (minimum validation loss) model so far
                torch.save(model, os.path.join(models_dir,f"AutoencoderTransformer.pt"))


if __name__ == "__main__":
    config = dotenv_values(".env")
    data_dir = config['DATA_ROOT']
    intermediate_dir = config['INTERMEDIATE_DIR']
    fingerprints_dir = config['FINGERPRINTS_DIR']
    models_dir = config['MODEL_DIR']
    max_pdbs = int(config['MAX_PDBS'])
    random_seed = int(config['RANDOM_SEED'])

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    if not os.path.isdir(fingerprints_dir):
        os.mkdir(fingerprints_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dim', default=5, type=int)
    parser.add_argument('-hi', '--hidden_dim', default=32, type=int)
    parser.add_argument('-l', '--num_layers', default=2, type=int)
    parser.add_argument('-nh', '--num_heads', default=5, type=int)
    parser.add_argument('-o', '--output_dim', default=256, type=int)
    parser.add_argument('-e', '--epochs', default=10, type=int)
    args = parser.parse_args()

    generate_neighbour_fingerprints()

    train(args.input_dim, args.hidden_dim, args.num_layers, args.num_heads, args.output_dim, args.epochs)
