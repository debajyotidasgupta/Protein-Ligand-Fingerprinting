import os
import requests
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from .utils import download_smiles_ligand_name, bitwise_or_fingerprint, download_smiles_pdb, download_smiles_cid, download_pdb


def get_ligands(pdb: str) -> list:
    """ Get ligands from PDB ID.

    - Utilizes the PUG REST API to get ligands from a PDB ID.
    - Returns a list of ligands.
    - Also returns the PubMed ID if available.

    Args:
        pdb (str): PDB ID

    Returns:
        list: List of ligands
    """
    URL = f'https://data.rcsb.org/rest/v1/core/entry/{pdb.lower()}'
    response = requests.get(URL)

    res = {}
    if response.status_code == 200:
        data = response.json()
        if 'rcsb_entry_container_identifiers' in data:
            if 'pubmed_id' in data['rcsb_entry_container_identifiers']:
                res['pubmed_id'] = data['rcsb_entry_container_identifiers']['pubmed_id']

        if 'rcsb_entry_info' in data:
            if 'nonpolymer_bound_components' in data['rcsb_entry_info']:
                res['ligands'] = data['rcsb_entry_info']['nonpolymer_bound_components']

    return res


# return rdkit's bitVector
def get_ligand_fingerprint_name(ligand: str, download_path: str) -> ExplicitBitVect:
    """ Get ligand fingerprint from ligand name.

    - Utilizes the download_smiles_ligand_name function to download the SMILES file.
    - Returns the fingerprint of the ligand.

    Args:
        ligand (str): Ligand name
        download_path (str): Path to download the SMILES file

    Returns:
        ExplicitBitVect: Fingerprint of the ligand

    Raises:
        FileNotFoundError: If the SMILES file is not found.
    """
    if not os.path.exists(os.path.join(download_path, f"{ligand}.smiles")):
        download_smiles_ligand_name(ligand, download_path=download_path)

    with open(os.path.join(download_path, f"{ligand}.smiles"), "r") as f:
        smiles = f.read().strip()

    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return fp


def get_ligand_fingerprint(pdb: str, smiles_path: str, pdb_path: str) -> ExplicitBitVect:
    """ Get ligand fingerprint from PDB ID.

    - Utilizes the get_ligands function to get the ligands from the PDB ID.
    - Returns the fingerprint of the ligands.

    Args:
        pdb (str): PDB ID
        smiles_path (str): Path to download the SMILES file
        pdb_path (str): Path to download the PDB file   

    Returns:
        ExplicitBitVect: Fingerprint of the ligands

    Raises:
        FileNotFoundError: If the SMILES file is not found.

    Example:
        >>> from fingerprint import get_ligand_fingerprint
        >>> get_ligand_fingerprint("TR8", "smiles", "pdb")
        <rdkit.DataStructs.cDataStructs.ExplicitBitVect object at 0x7f8b1c0b0e00>
    """
    info = get_ligands(pdb)

    fingerprint = None

    try:
        for ligand in info['ligands']:  # list of ligands
            fp = get_ligand_fingerprint_name(
                ligand, smiles_path)  # get fingerprint of ligand
            if fingerprint is None:  # if fingerprint is None, set it to the fingerprint of the ligand
                fingerprint = fp    # set fingerprint to the fingerprint of the ligand
            else:                  # else, bitwise or the fingerprint with the fingerprint of the ligand
                # bitwise or the fingerprint with the fingerprint of the ligand
                fingerprint = bitwise_or_fingerprint(fingerprint, fp)
    except:
        try:
            download_smiles_pdb(pdb, download_path=smiles_path)
            with open(os.path.join(smiles_path, f"{pdb}.smiles"), "r") as f:
                smiles = f.read().strip()
            mol = Chem.MolFromSmiles(smiles)
        except:
            try:
                download_smiles_cid(info['pubmed_id'],
                                    download_path=smiles_path)
                with open(os.path.join(smiles_path, f"{info['pubmed_id']}.smiles"), "r") as f:
                    smiles = f.read().strip()
                mol = Chem.MolFromSmiles(smiles)
            except:
                download_pdb(pdb, pdb_path)
                mol = Chem.MolFromPDBFile(os.path.join(pdb_path, f"{pdb}.pdb"))

        fingerprint = MACCSkeys.GenMACCSKeys(mol)

    return fingerprint
