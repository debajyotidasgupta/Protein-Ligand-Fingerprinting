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
            'C': 2,
            'O': 3,
            'R': 4,
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
                # Get the atom name
                atom_name = neighbour.get_atom_name()

                # Get the index
                index = self.coding.get(atom_name, 4)

                # Update the fingerprint
                fingerprints[atom.get_atom_id()][index] += 1
