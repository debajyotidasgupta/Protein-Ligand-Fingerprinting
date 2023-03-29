from .parser import Atom


def euclidean_distance(atom1: Atom, atom2: Atom) -> float:
    """Calculate the Euclidean distance between two atoms.

    Args:
        atom1 (Atom): The first atom.
        atom2 (Atom): The second atom.

    Returns:
        float: The Euclidean distance between the two atoms.
    """

    # Calculate the Euclidean distance between the two atoms
    distance = (
        (atom1.x - atom2.x) ** 2
        + (atom1.y - atom2.y) ** 2
        + (atom1.z - atom2.z) ** 2
    ) ** 0.5

    # Return the Euclidean distance
    return distance
