from typing import List, Dict, Union, Any
from .alphabets import get_alphabet, FULL_ALPHABETS
from .parser import Atom, Protein
from scipy.spatial import cKDTree

import os
import requests
import numpy as np
import pandas as pd
from collections.abc import Sequence


def load_smiles(smiles_path=None):
    if os.path.exists(smiles_path):
        with open(smiles_path, "r") as f:
            smiles = f.read().strip()
        print(f"Found SMILES at {smiles_path}")
        return smiles
    else:
        raise FileNotFoundError(
            f"SMILES file not found for {pdb_id} at {smiles_path}.")


def download_smiles(pdb_id, download_path=None):
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


def protein2seq(protein: Protein) -> str:
    """Convert a protein object into a sequence string.

    Args:
        protein (Protein): The protein object.

    Returns:
        str: The sequence string.
    """
    sequence = ""
    for atom in protein.atoms:
        if atom.get_atom_name() == "CA":
            sequence += IUPAC_3_to_1[atom.get_residue_name()]
    return sequence
