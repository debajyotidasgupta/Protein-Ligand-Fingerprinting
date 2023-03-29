from .base import *
from .utils import *
import numpy as np
from tqdm import tqdm


class NeighbourFingerprint(BaseFingerprint):
    def __init__(self, pdb, *args, **kwargs):
        super(NeighbourFingerprint, self).__init__(pdb, *args, **kwargs)

        self.coding = {
            'N': 0,
            'CA': 1,
            'C': 2,
            'O': 3,
            'R': 4,
        }

        self.fingerprint = self._get_fingerprint()

    def _get_fingerprint(self, distance=7.0):
        # Get the protein
        protein = self.get_pdb().protein

        # Get the atoms
        atoms = protein.atoms

        # Create fingerprint
        fingerprints = np.zeros((len(protein.atoms), 5))

        # Iterate over the atoms
        for pos, atom in tqdm(list(enumerate(atoms)), desc='Parsing Neighbours'):
            # Get the neighbours of the atom
            neighbours = distance_search(atom, protein, distance)

            # Iterate over the neighbours
            for neighbour in neighbours:
                # Get the atom name
                atom_name = neighbour.get_atom_name()

                # Get the index
                index = self.coding.get(atom_name, 4)

                # Update the fingerprint
                fingerprints[pos][index] += 1

        return fingerprints
