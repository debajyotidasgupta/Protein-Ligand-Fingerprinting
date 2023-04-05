from typing import List, Dict, Union, Any
from .alphabets import get_alphabet, FULL_ALPHABETS
from .parser import Atom, Protein
from .transformer import AutoencoderTransformer
from scipy.spatial import cKDTree

import os
import requests
import numpy as np
import pandas as pd
from collections.abc import Sequence
import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


def fingerprint_to_bitarray(fp):
    return np.array([fp.GetBit(i) for i in range(fp.GetNumBits())], dtype=np.uint8)


def bitarray_to_fingerprint(bitarray, fp):
    new_fp = ExplicitBitVect(fp.GetNumBits())
    for i, bit in enumerate(bitarray):
        if bit:
            new_fp.SetBit(i)
    return new_fp


def bitwise_or_fingerprint(fp1: ExplicitBitVect, fp2: ExplicitBitVect) -> ExplicitBitVect:
    """Bitwise OR operation on two fingerprints.

    Parameters
    ----------
    fp1 : ExplicitBitVect
        First fingerprint.
    fp2 : ExplicitBitVect
        Second fingerprint.

    Returns
    -------
    ExplicitBitVect
        Bitwise OR of two fingerprints.
    """
    bitarray1 = fingerprint_to_bitarray(fp1)
    bitarray2 = fingerprint_to_bitarray(fp2)

    bitarray_or = np.bitwise_or(bitarray1, bitarray2)
    fp_or = bitarray_to_fingerprint(bitarray_or, fp1)

    return fp_or


def load_smiles(smiles_path=None):
    if os.path.exists(smiles_path):
        with open(smiles_path, "r") as f:
            smiles = f.read().strip()
        print(f"Found SMILES at {smiles_path}")
        return smiles
    else:
        raise FileNotFoundError(
            f"SMILES file not found at {smiles_path}.")


def download_smiles_pdb(pdb_id, download_path=None):
    if os.path.exists(os.path.join(download_path, f"{pdb_id}.smiles")):
        print(f"Found SMILES for {pdb_id}")
        return

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{pdb_id}/property/IsomericSMILES/TXT"
    response = requests.get(url)

    if response.status_code == 200:
        smiles = response.text.strip()
        print(f"Found SMILES for {pdb_id}")
        # Save the SMILES
        if download_path is None:
            download_path = os.getcwd()
        with open(os.path.join(download_path, f"{pdb_id}.smiles"), "w") as f:
            f.write(smiles)
    else:
        raise ValueError(f"Could not find SMILES for PDB ID: {pdb_id}")


def download_smiles_cid(cid, download_path=None):
    if os.path.exists(os.path.join(download_path, f"{cid}.smiles")):
        print(f"Found SMILES for {cid}")
        return

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/TXT"
    response = requests.get(url)

    if response.status_code == 200:
        smiles = response.text.strip()
        print(f"Found SMILES for {cid}")
        # Save the SMILES
        if download_path is None:
            download_path = os.getcwd()
        with open(os.path.join(download_path, f"{cid}.smiles"), "w") as f:
            f.write(smiles)
    else:
        raise ValueError(f"Could not find SMILES for CID: {cid}")


def download_smiles_ligand_name(name: str, download_path=None):
    if os.path.exists(os.path.join(download_path, f"{name}.smiles")):
        print(f"Found SMILES for {name}")
        return

    url = f'https://data.rcsb.org/rest/v1/core/chemcomp/{name.lower()}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if 'rcsb_chem_comp_descriptor' in data:
            if 'smiles' in data['rcsb_chem_comp_descriptor']:
                smiles = data['rcsb_chem_comp_descriptor']['smiles']
                print(f"Found SMILES for {name}")
                with open(os.path.join(download_path, f"{name}.smiles"), "w") as f:
                    f.write(smiles)
                return
    raise ValueError(f"Could not find SMILES for ligand name: {name}")


def download_pdb(pdb_id, download_path=None):
    if os.path.exists(os.path.join(download_path, f"{pdb_id}.pdb")):
        print(f"Found PDB file for {pdb_id} as {pdb_id}.pdb")
        return

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    response = requests.get(url)

    if response.status_code == 200:
        pdb_content = response.text
        if download_path is None:
            download_path = os.getcwd()
        with open(os.path.join(download_path, f"{pdb_id}.pdb"), "w") as f:
            f.write(pdb_content)
        print(f"Downloaded PDB file for {pdb_id} as {pdb_id}.pdb")
    else:
        raise ValueError(f"Could not download PDB file for PDB ID: {pdb_id}")


def check_list(array: Any) -> bool:
    """Check whether input is a sequence, list, or array-like object.
    Parameters
    ----------
    array : type
        Input array (or any parameter type).
    Returns
    -------
    bool
        Returns True if input is a list/sequence, array, or Series.
    """
    if not isinstance(array, (Sequence, np.ndarray, pd.Series)):
        return False
    return True


def get_residues_near_atom(
    query: Atom,
    database: Protein,
    threshold: float,
) -> List[Atom]:
    """Search for atoms in the database that are within a certain distance of the query.

    Args:
        query (Atom): The query atom.
        database (Protein): The database to search.
        threshold (float): The distance threshold.

    Returns:
        List[Atom]: A list of atoms that are within the distance threshold.
    """

    # Get the coordinates of the query
    query_coords = query.get_coordinates()

    # Get the coordinates of the database
    database_coords = database.get_coordinates()

    # Create a KDTree
    tree = cKDTree(database_coords)

    # Query the KDTree
    neighbours = tree.query_ball_point(query_coords, threshold)

    # Get the atoms
    atoms = [database.atoms[i] for i in neighbours]

    return atoms


def reduce(
    sequence: str, alphabet: Union[str, int], mapping: dict = FULL_ALPHABETS
) -> str:
    """Reduce sequence into character space of specified alphabet.
    Parameters
    ----------
    sequence : str
        Input sequence.
    alphabet : Union[str, int]
        Alphabet name or number (see `snekmer.alphabet`).
    mapping : dict
        Defined mapping for alphabet (the default is FULL_ALPHABETS).
    Returns
    -------
    str
        Transformed sequence.
    """
    sequence = str(sequence).rstrip("*")
    alphabet_map: Dict[str, str] = get_alphabet(alphabet, mapping=mapping)
    return sequence.translate(sequence.maketrans(alphabet_map))


IUPAC_3_to_1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


def pdb_seq(pdb: str, pdb_path: str) -> str:
    """Extracts the sequence from a pdb file.
    Args:
        pdb: pdb id
        pdb_path: path to the pdb file
    Returns:
        seq: sequence of the pdb file
    """
    download_pdb(pdb, pdb_path)

    print(f"Extracting sequence from {pdb}.pdb")
    seq = ''
    with open(os.path.join(pdb_path, f"{pdb}.pdb"), 'r') as f:
        for line in f:
            if line.startswith('SEQRES'):
                AAs = line[19:].split()
                for aa in AAs:
                    if aa in IUPAC_3_to_1.keys():
                        seq += IUPAC_3_to_1[aa]
    return seq


def encode(fingerprints, model_path):
    model = torch.load(model_path)
    reduce_dim = False

    if isinstance(fingerprints, np.ndarray) and fingerprints.ndim == 2:
        fingerprints = [fingerprints]
        reduce_dim = True

    data_loader = [torch.from_numpy(x) for x in fingerprints]
    data_loader = [torch.unsqueeze(x, 0) for x in data_loader]
    data_loader = [x.to(torch.float) for x in data_loader]

    model.eval()

    outputs = []

    for i, data in enumerate(data_loader, 0):
        input = data
        output, _ = model(input)
        outputs.append(torch.squeeze(output).detach().numpy())

    if reduce_dim:
        outputs = outputs[0]

    return outputs
