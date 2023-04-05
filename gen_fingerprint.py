import os
import sys
import json
import gdown
import numpy as np
from dotenv import dotenv_values
from fingerprint import *

if __name__ == '__main__':
    pdb = sys.argv[1].strip().split('.')[0]

    config = dotenv_values(".env")
    PDB_DATA = config['PDB_DATA']
    SMILES_DATA = config['SMILES_DATA']
    OUTPUT_DATA = config['OUTPUT_DATA']
    MODEL_PATH = config['MODEL_PATH']

    # if output dir with name of PDB ID does not exist, create it
    if not os.path.exists(os.path.join(OUTPUT_DATA, pdb)):
        os.mkdir(os.path.join(OUTPUT_DATA, pdb))

    print("""\033[2J
    ┌──────────────────────────────────────────────┐
    │                                              │
    │         Protein Ligand Fingerprints          │
    │                                              │
    ┠──────────────────────────────────────────────┨
    │                                              │
    │  This script generates the protein ligand    │
    │  fingerprints for the given PDB ID.          │
    │                                              │
    │  Author:                                     │
    │  - Debajyoti Dasgupta [18CS30051]            │
    │    (debajyotidasgupta6@gmail.com)            │
    │  - Somnath Jena [18CS30047]                  │
    │    (somnathjena.2011@gmail.com)              │
    │                                              │
    │  Subject: Computational Biopysics            │
    │           Algorithms to Applications         │
    │           (CS61060)                          │
    |                                              |
    │  Date: 2023-04-05                            │
    │  Semester: 2022-23 Autumn                    │
    │                                              │
    ┠──────────────────────────────────────────────┨
    │  Disclaimer:                                 │
    │  - This script is for academic purposes      │
    │    only.                                     │
    │  - The authors are not responsible for any   │
    │    misuse of this script.                    │
    │  - The authors are not responsible for any   │
    │    damage, loss of data, or any other        │
    │    consequences of using this script.        │
    │                                              │
    └──────────────────────────────────────────────┘
    \n
    ──────────────────────────────────────
    Note:
    - For changes to parameters, please
      edit the .env file's values.
    - Read the README.md file for more
    ──────────────────────────────────────
    """)

    print("""
    ┌──────────────────────────────────────────────┐
    │                                              │
    │    Protein Ligand Neighbour Fingerprint      │
    │                                              │
    ┠──────────────────────────────────────────────┨
    │                                              │
    │  For each atom in the ligand, find the       │
    │  protein atoms within a cutoff distance      │
    │  The fingerprint is a frequency vector       │
    │  of the {N, C, CA, R, O} atoms in the        │
    │  neighbourhood of the ligand atoms. R        │
    │  represents the atoms not part of the        │
    │  backbone and is represented by the          │
    │  centroid of the side chain.                 │
    │                                              │
    └──────────────────────────────────────────────┘
    \n
    """)

    # Create a new instance of the class
    protein_ligand_complex = ProteinLigandSideChainComplex()

    # Download the PDB file
    download_pdb(pdb, PDB_DATA)

    # Load the PDB file
    protein_ligand_complex.load_pdb(os.path.join(PDB_DATA, f'{pdb}.pdb'))

    # Print the protein fingerprint
    fingerprint = NeighbourFingerprint(protein_ligand_complex, distance=10.)
    fingerprint = fingerprint.get_fingerprint()
    print(f"\nFingerprint shape = {fingerprint.shape}")
    print(f'Fingerprint:\n{fingerprint}')

    print(
        f"\n═════▷ Saving fingerprint to {os.path.join(OUTPUT_DATA, pdb, f'{pdb}_neighbour.txt')} ◁═════")
    np.savetxt(os.path.join(OUTPUT_DATA, pdb,
               f'{pdb}_neighbour.txt'), fingerprint, fmt='%d')

    print(f"\n────────────────────────────────────────────────────\n")

    print(f"""
    ┌──────────────────────────────────────────────┐
    │                                              │
    │    Protein Ligand Neighbour Fingerprint      │
    │            (Transformer Encoded)             │
    │                                              │
    ┠──────────────────────────────────────────────┨
    │                                              │
    │  Use a pre-trained transformer (explained    │
    │  in the README) to encode the fingerprint    │
    │  to a constant length feature vector.        │
    │                                              │
    └──────────────────────────────────────────────┘
    \n
    """)

    try:
        url = config['MODEL_URL']
        output = config['MODEL_PATH']

        if os.path.exists(output):
            print(f"Model already downloaded to {output}")
        else:
            gdown.download(url, output, quiet=False)

        # encode the fingerprint using transformers model to get a constant length feature vector
        fingerprint = encode(fingerprint, MODEL_PATH)
        print(f"Fingerprint shape = {fingerprint.shape}")
        print(f"Fingerprint:\n{np.array2string(fingerprint, threshold=100)}")

        print(
            f"\n═════▷ Saving fingerprint to {os.path.join(OUTPUT_DATA, pdb, f'{pdb}_neighbour_transformer.txt')} ◁═════")
        np.savetxt(os.path.join(OUTPUT_DATA, pdb,
                                f'{pdb}_neighbour_transformer.txt'), fingerprint)
    except Exception as e:
        print(
            f"═════▷ Skipping transformer encoding due to error: {e}\n  ◁═════")

    print(f"\n────────────────────────────────────────────────────\n")

    print(f"""
    ┌──────────────────────────────────────────────┐
    │                                              │
    │       AAR Kmer based Fingerprint             │
    │                                              │
    ┠──────────────────────────────────────────────┨
    │                                              │
    │  Amino Acid Recoding (AAR) is a method       │
    │  to convert the amino acid sequence into a   │
    │  sequence of feature based on the chemical   │
    │  properties of the amino acids.              │
    │                                              │
    │  The fingerprint is a vector of the number   │
    │  of times each of the k-mers appear in the   │
    │  AAR sequence.                               │
    │                                              │
    └──────────────────────────────────────────────┘
    \n
    """)
    # Print the protein fingerprint
    seq = pdb_seq(pdb, PDB_DATA)
    vec = KmerVec(int(config['AAR_TYPE_NUM']), int(config['KMER_SIZE']))
    vec.set_kmer_set()
    fingerprint = vec.reduce_vectorize(seq)
    print(
        f"\nAAR type = {int(config['AAR_TYPE_NUM'])} | Kmer size = {config['KMER_SIZE']}")
    print(f"Fingerprint length = {len(fingerprint)}")
    print(f'Fingerprint:\n{json.dumps(fingerprint, indent=4)}')

    print(
        f"\n═════▷ Saving fingerprint to {os.path.join(OUTPUT_DATA, pdb, f'{pdb}_aar_kmer.json')} ◁═════")
    with open(os.path.join(OUTPUT_DATA, pdb, f'{pdb}_aar_kmer.json'), 'w') as f:
        json.dump(fingerprint, f, indent=4)

    print(f"\n────────────────────────────────────────────────────\n")

    print(f"""
    ┌──────────────────────────────────────────────┐
    │                                              │
    │       Ligand MACCS KEY Fingerprint           │
    │                                              │
    ┠──────────────────────────────────────────────┨
    │                                              │
    │  The MACCS keys are a set of 166 binary      │
    │  molecular descriptors that are used to      │
    │  characterize the chemical structure of      │
    │  a molecule.                                 │
    │                                              │
    │  The fingerprint is a vector of the number   │
    │  of times each of the MACCS keys appear in   │
    │  the ligand.                                 │
    │                                              │
    └──────────────────────────────────────────────┘
    \n
    """)

    # Print the ligand Fingerprint
    fp = get_ligand_fingerprint(pdb=pdb,
                                smiles_path=SMILES_DATA,
                                pdb_path=PDB_DATA)
    fingerprint = fp.ToBitString()
    print(f"Fingerprint length = {len(fingerprint)}")
    print(f'Fingerprint:\n{fingerprint}')

    print(
        f"\n═════▷ Saving fingerprint to {os.path.join(OUTPUT_DATA, pdb, f'{pdb}_maccs.txt')} ◁═════")
    with open(os.path.join(OUTPUT_DATA, pdb, f'{pdb}_maccs.txt'), 'w') as f:
        f.write(fingerprint)

    print(f"\n────────────────────────────────────────────────────\n")

    print(f"""
    ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
    🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳
    ┌──────────────────────────────────────────────┐
    │                                              │
    │        PROGRAM EXECUTION COMPLETED           │
    │               SUCCESSFULLY!                  │
    │                                              │
    └──────────────────────────────────────────────┘
    \n
    """)
