import numpy as np


class BaseFingerprint:
    """Base class for fingerprint generation.

    Parameters
    ----------
    pdb : ProteinLigandComplex
        Protein-ligand complex object.

    Returns
    -------
    None
    """

    def __init__(self, pdb, *args, **kwargs):
        """Constructor method.

        Parameters
        ----------
        pdb : ProteinLigandComplex
            Protein-ligand complex object.  

        Returns
        -------
        None
        """
        self.pdb = pdb
        self.fingerprint = None

    def _get_fingerprint(self):
        """Method to generate the fingerprint.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        raise NotImplementedError

    def get_fingerprint(self):
        """Method to get the fingerprint.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        return self.fingerprint

    def get_pdb(self):
        """Method to get the protein-ligand complex object.

        Parameters
        ----------
        None

        Returns
        -------
        ProteinLigandComplex
            Protein-ligand complex object.
        """

        return self.pdb

    def get_fingerprint_length(self):
        """Method to get the length of the fingerprint.

        Parameters
        ----------
        None    

        Returns
        -------
        int
            Length of the fingerprint.
        """

        return len(self.fingerprint)

    def save_fingerprint(self, file_path):
        """Save the fingerprint to a file.

        Parameters
        ----------
        file_path : str
            Path to the file.

        Returns
        -------
        None
        """
        raise NotImplementedError
