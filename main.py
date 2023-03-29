from fingerprint import *

if __name__ == '__main__':
    # Create a new instance of the class
    protein_ligand_complex = ProteinLigandComplex()

    # Load the PDB file
    protein_ligand_complex.load_pdb("data/2xni.pdb")

    # Print the protein fingerprint
    fingerprint = NeighbourFingerprint(protein_ligand_complex)
    print(fingerprint.get_fingerprint())
