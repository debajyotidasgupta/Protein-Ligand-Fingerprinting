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
        return f"{self.atom_name}:{self.residue_name}:{self.chain_id}{self.residue_id}:{self.x},{self.y},{self.z}"


class Protein:
    def __init__(self):
        self.atoms = []
        self.chains = {}

    def add_atom(self, atom):
        self.atoms.append(atom)
        chain = self.chains.get(atom.chain_id, [])
        chain.append(atom)
        self.chains[atom.chain_id] = chain


class Ligand:
    def __init__(self):
        self.atoms = []

    def add_atom(self, atom):
        self.atoms.append(atom)


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

                atom = Atom(atom_id, atom_name, residue_name, chain_id, residue_id, x, y, z)

                if line.startswith("ATOM"):
                    self.protein.add_atom(atom)
                else:
                    self.ligand.add_atom(atom)

    def __repr__(self):
        return f"ProteinLigandComplex with {len(self.protein.atoms)} protein atoms and {len(self.ligand.atoms)} ligand atoms."
