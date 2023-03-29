class BaseFingerprint:
    def __init__(self, pdb, *args, **kwargs):
        self.pdb = pdb
        self.fingerprint = None

    def _get_fingerprint(self):
        raise NotImplementedError

    def get_fingerprint(self):
        return self.fingerprint

    def get_pdb(self):
        return self.pdb

    def get_fingerprint_length(self):
        return len(self.fingerprint)
