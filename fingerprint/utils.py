from typing import List
from .parser import Atom, Protein
from scipy.spatial import cKDTree


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
