from fingerprint.parser import *

if __name__ == '__main__':
    # Create a new instance of the class
    protein_ligand_complex = ProteinLigandComplex()

    # Load the PDB file
    protein_ligand_complex.load_pdb("data/2xni.pdb")

    for atom in protein_ligand_complex.protein.atoms[:20]:
        print(atom)
