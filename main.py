from fingerprint import *

from argparse import ArgumentParser

if __name__ == '__main__':

    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--pdb", type=str, help="The PDB file to be used for the fingerprint generation.")
    arg_parser.add_argument(
        "--output", type=str, help="The output file to save the fingerprint.")
    args = arg_parser.parse_args()

    # Create a new instance of the class
    protein_ligand_complex = ProteinLigandSideChainComplex()

    # Load the PDB file
    protein_ligand_complex.load_pdb(args.pdb)

    # Print the protein fingerprint
    fingerprint = NeighbourFingerprint(protein_ligand_complex)
    print(fingerprint.get_fingerprint())

    # Save the fingerprint
    fingerprint.save_fingerprint(args.output)
