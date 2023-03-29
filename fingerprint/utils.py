from typing import List
from parser import Atom, Protein
from distance import euclidean_distance


def distance_search(
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

    # Initialize the list of atoms that are within the distance threshold
    atoms_within_threshold = []

    # Loop over all atoms in the database
    for atom in database.atoms:

        # Calculate the distance between the query and the current atom
        distance = euclidean_distance(query, atom)

        # If the distance is less than the threshold, add the atom to the list
        if distance < threshold:
            atoms_within_threshold.append(atom)

    # Return the list of atoms that are within the distance threshold
    return atoms_within_threshold
