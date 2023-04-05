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
        return self.pdb

    def get_fingerprint_length(self):
        return len(self.fingerprint)

    def save_fingerprint(self, file_path):
        raise NotImplementedError
