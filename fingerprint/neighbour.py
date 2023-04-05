from .base import *
from .utils import *
from .parser import *
from tqdm import tqdm
import math


class NeighbourFingerprint(BaseFingerprint):
    def __init__(self, pdb: ProteinLigandComplex, *args, **kwargs):
        """The NeighbourFingerprint class is used to generate a fingerprint of the protein-ligand complex.
        Args:
            pdb (ProteinLigandComplex): The protein-ligand complex.

        Returns:
            None
        """
        super(NeighbourFingerprint, self).__init__(pdb, *args, **kwargs)

        # Define the coding scheme
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

        # Get the atoms of ligand
        atoms = self.get_pdb().ligand.get_atoms()

        # Create fingerprint
        fingerprints = np.zeros((len(protein.atoms), 5))

        # Iterate over the atoms
        for pos, atom in tqdm(list(enumerate(atoms)), desc='Parsing Neighbours'):
            # Get the neighbours of the atom
            neighbours = get_residues_near_atom(atom, protein, distance)

            # Iterate over the neighbours
            for neighbour in neighbours:
                # Get the atom name
                atom_name = neighbour.get_atom_name()

                # Get the index
                index = self.coding.get(atom_name, 4)

                # Update the fingerprint
                fingerprints[pos][index] += 1

        return fingerprints

    def get_fingerprint(self):
        return self.fingerprint

    def save_fingerprint(self, file_path):
        np.savetxt(file_path, self.fingerprint, fmt='%1.8f')
