from base import *
from utils import *
import numpy as np


class NeighbourFingerprint(BaseFingerprint):
    def __init__(self, pdb, *args, **kwargs):
        super(NeighbourFingerprint, self).__init__(pdb, *args, **kwargs)
        self.fingerprint = self._get_fingerprint()

        self.coding = {
            'N': 0,
            'CA': 1,
            'C': 1,
            'O': 2,
            'CB': 3,
            'CG': 4,
            'CD': 5,
            'CE': 6,
            'NZ': 7,
            'SG': 8,
        }

    def _get_fingerprint(self, distance=7.0):
        # Get the list of atoms
        atoms = self.get_pdb().get_atoms()

        # Create fingerprint
        fingerprints = np.zeros((len(atoms), 5))

        # Iterate over the atoms
        for atom in atoms:
            # Get the neighbours of the atom
            neighbours = distance_search(atom, atoms, distance)

            # Iterate over the neighbours
            for neighbour in neighbours:
                # Get the distance between the atom and the neighbour
                distance = get_distance(atom, neighbour)

                # Get the angle between the atom and the neighbour
                angle = get_angle(atom, neighbour)

                # Get the dihedral angle between the atom and the neighbour
                dihedral = get_dihedral(atom, neighbour)

                # Get the torsion angle between the atom and the neighbour
                torsion = get_torsion(atom, neighbour)

                # Get the index of the neighbour
                neighbour_index = atoms.index(neighbour)

                # Update the fingerprint
                fingerprints[neighbour_index] = [
                    distance, angle, dihedral, torsion, 1]
