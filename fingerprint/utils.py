from typing import List, Dict, Union, Any
from .alphabets import get_alphabet, FULL_ALPHABETS
from .parser import Atom, Protein
from scipy.spatial import cKDTree

import numpy as np
import pandas as pd
from collections.abc import Sequence


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
