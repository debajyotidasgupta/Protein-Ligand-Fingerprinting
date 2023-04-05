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
    if not os.path.exists(os.path.join(download_path, f"{ligand}.smiles")):
        download_smiles_ligand_name(ligand, download_path=download_path)

    with open(os.path.join(download_path, f"{ligand}.smiles"), "r") as f:
        smiles = f.read().strip()

    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return fp


def get_ligand_fingerprint(pdb: str, smiles_path: str, pdb_path: str) -> ExplicitBitVect:
    info = get_ligands(pdb)

    fingerprint = None

    try:
        for ligand in info['ligands']:
            fp = get_ligand_fingerprint_name(ligand, smiles_path)
            if fingerprint is None:
                fingerprint = fp
            else:
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
