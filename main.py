import os
import sys
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from fingerprint import *

if __name__ == '__main__':
    pdb = sys.argv[1].strip().split('.')[0]

    PDB_DATA = './data/PDB/'
    SMILES_DATA = './data/SMILES/'
    OUTPUT_DATA = './output/'

    # Create a new instance of the class
    protein_ligand_complex = ProteinLigandSideChainComplex()

    # Download the PDB file
    download_pdb(pdb, PDB_DATA)

    # Load the PDB file
    protein_ligand_complex.load_pdb(os.path.join(PDB_DATA, f'{pdb}.pdb'))

    # # Print the protein fingerprint
    fingerprint = NeighbourFingerprint(protein_ligand_complex)
    print(fingerprint.get_fingerprint())

    # Save the fingerprint
    fingerprint.save_fingerprint(
        os.path.join(OUTPUT_DATA, f'{pdb}.txt'))

    # Print the protein
    seq = protein2seq(protein_ligand_complex.protein)
    vec = KmerVec(0, k=2)
    vec.set_kmer_set()
    print(vec.vectorize(seq))

    # Print the ligand Fingerprint
    try:
        # Download the SMILES file
        download_smiles(pdb, SMILES_DATA)

        # Load the SMILES file
        smiles = load_smiles(os.path.join(SMILES_DATA, f'{pdb}.smiles'))

        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = Chem.MolFromPDBFile(os.path.join(PDB_DATA, f'{pdb}.pdb'))
    fp = MACCSkeys.GenMACCSKeys(mol)
    print(fp.ToBitString())
