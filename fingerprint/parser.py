import numpy as np


class Atom:
    def __init__(self, atom_id, atom_name, residue_name, chain_id, residue_id, x, y, z):
        self.atom_id = atom_id
        self.atom_name = atom_name
        self.residue_name = residue_name
        self.chain_id = chain_id
        self.residue_id = residue_id
        self.x = x
        self.y = y
        self.z = z

    def get_atom_name(self):
        return self.atom_name

    def get_residue_name(self):
        return self.residue_name

    def get_coordinates(self):
        return np.array([self.x, self.y, self.z])

    def __repr__(self):
        return f"ATOM: \n\
        atom_id: {self.atom_id} \n\
        atom_name: {self.atom_name} \n\
        residue_name: {self.residue_name} \n\
        chain_id: {self.chain_id} \n\
        residue_id: {self.residue_id} \n\
        x: {self.x} \n\
        y: {self.y} \n\
        z: {self.z} \n\
        "


class Protein:
    def __init__(self):
        self.atoms = []
        self.chains = {}

    def add_atom(self, atom):
        self.atoms.append(atom)
        chain = self.chains.get(atom.chain_id, [])
        chain.append(atom)
        self.chains[atom.chain_id] = chain

    def get_atoms(self):
        return self.atoms

    def get_coordinates(self):
        coords = []
        for atom in self.atoms:
            coords.append(atom.get_coordinates())
        return np.array(coords)

    def __repr__(self) -> str:
        return f"Protein: \n\
        atoms: {len(self.atoms)} \n\
        chains: {len(self.chains)} \n\
        "


class Ligand:
    def __init__(self):
        self.atoms = []

    def add_atom(self, atom):
        self.atoms.append(atom)

    def get_atoms(self):
        return self.atoms

    def get_coordinates(self):
        coords = []
        for atom in self.atoms:
            coords.append(atom.get_coordinates())
        return np.array(coords)

    def __repr__(self) -> str:
        return f"Ligand: \n\
        atoms: {len(self.atoms)} \n\
        "


class ProteinLigandSideChainComplex:
    def __init__(self):
        self.protein = Protein()
        self.ligand = Ligand()

    def _compute_centroid(self, side_chain_atoms):
        x = 0.0
        y = 0.0
        z = 0.0
        atom_id = -1
        for atom in side_chain_atoms:
            x += atom.x
            y += atom.y
            z += atom.z
            if atom.atom_name == "CB":
                atom_id = atom.atom_id
        n_atoms = len(side_chain_atoms)
        return (x/n_atoms, y/n_atoms, z/n_atoms, atom_id)

    def load_pdb(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()

        side_chain_atoms_dict = {}

        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_id = int(line[6:11])
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain_id = line[21]
                residue_id = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                atom = Atom(atom_id, atom_name, residue_name,
                            chain_id, residue_id, x, y, z)

                is_atom = False
                if line.startswith("ATOM"):
                    is_atom = True

                if is_atom and atom_name.strip()[0] not in ['N', 'C', 'O', 'CA']:
                    if chain_id not in side_chain_atoms_dict:
                        side_chain_atoms_dict[chain_id] = {}
                    if residue_id not in side_chain_atoms_dict[chain_id]:
                        side_chain_atoms_dict[chain_id][residue_id] = []
                    side_chain_atoms_dict[chain_id][residue_id].append(atom)
                else:
                    if is_atom:
                        self.protein.add_atom(atom)
                    else:
                        self.ligand.add_atom(atom)

        side_chain_representative_atoms = []
        for chain_id in side_chain_atoms_dict.keys():
            for residue_id in side_chain_atoms_dict[chain_id].keys():
                side_chain_atoms = side_chain_atoms_dict[chain_id][residue_id]
                x, y, z, atom_id = self._compute_centroid(side_chain_atoms)
                if atom_id >= 0:
                    atom = Atom(atom_id, "R", residue_name,
                                chain_id, residue_id, x, y, z)
                    side_chain_representative_atoms.append(atom)

        def sorter(x): return x.atom_id
        self.protein.atoms = sorted(
            self.protein.atoms + side_chain_representative_atoms, key=sorter)

    def __repr__(self):
        return f"ProteinLigandComplex: \n\
        protein: {self.protein} \n\
        ligand: {self.ligand} \n\
        "


class ProteinLigandComplex:
    def __init__(self):
        self.protein = Protein()
        self.ligand = Ligand()

    def load_pdb(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_id = int(line[6:11])
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain_id = line[21]
                residue_id = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                atom = Atom(atom_id, atom_name, residue_name,
                            chain_id, residue_id, x, y, z)

                if line.startswith("ATOM"):
                    self.protein.add_atom(atom)
                else:
                    self.ligand.add_atom(atom)

    def __repr__(self):
        return f"ProteinLigandComplex: \n\
        protein: {self.protein} \n\
        ligand: {self.ligand} \n\
        "
