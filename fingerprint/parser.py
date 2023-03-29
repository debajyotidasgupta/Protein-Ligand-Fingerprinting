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

    def __repr__(self) -> str:
        return f"Ligand: \n\
        atoms: {len(self.atoms)} \n\
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
