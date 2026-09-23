"""Gamma-point CSR reader extracted from read_abacus.py.

Original authors: ChangWei Zhang and Yang Zhong.
Only the H/S reader required by wfc_dft.py is included.
"""

import re
import numpy as np
from scipy.sparse import csr_matrix as csr

ry2ha = 13.60580 / 27.21138506


class ABACUSHS:
    def __init__(self, file: str) -> None:
        """
        Initializes the ABACUSHS object by reading the data from the provided file.

        Args:
            file (str): The file containing the ABACUSHS data.
        """
        self.fp = open(file)
        line = self.fp.readline()  # Read first line to determine the format
        if 'STEP' in line:
            self.no_u = int(self.fp.readline().split()[-1])
        else:
            self.no_u = int(line.split()[-1])  # Number of orbitals in the unit cell.
        self.ncell_shift = int(self.fp.readline().split()[-1])

    def getHK(self, stru, k: np.ndarray = np.array([0, 0, 0]), isH: bool = False, isSOC: bool = False):
        """
        Returns the Hamiltonian matrix for the specified k-point.

        Args:
            stru (STRU): The structure object containing atomic information.
            k (np.ndarray, optional): The k-point for which to calculate the Hamiltonian, defaults to [0,0,0].
            isH (bool, optional): If True, scales the Hamiltonian by `ry2ha`, defaults to False.
            isSOC (bool, optional): If True, includes spin-orbit coupling, defaults to False.

        Returns:
            np.ndarray: The Hamiltonian matrix for the specified k-point.
        """
        assert np.all(k == 0)  # Only support gamma point

        dtype = np.float32 if not isSOC else np.complex64
        HK = np.zeros([self.no_u, self.no_u], dtype=dtype)

        while True:
            line = self.fp.readline()
            if not line:
                break
            tmp = line.split()
            cx, cy, cz = int(tmp[0]), int(tmp[1]), int(tmp[2])
            nh = int(tmp[3])
            if nh == 0:
                continue
            val = self.fp.readline()
            col = self.fp.readline().split()
            row = self.fp.readline().split()

            # Handle Hamiltonian values
            if not isSOC:
                val = list(map(float, val.split()))
            else:
                val_raw = re.findall(r'[\-\+\d\.eE]+', val)
                val_raw = np.asarray(val_raw, dtype=np.float32)
                val = np.zeros(len(val_raw) // 2, dtype=np.complex64)
                val += val_raw[0::2] + 1j * val_raw[1::2]

            col = list(map(int, col))
            row = list(map(int, row))
            hamilt = csr((val, col, row), shape=[self.no_u, self.no_u], dtype=dtype)
            if isH:
                hamilt *= ry2ha

            HK += hamilt

        return HK

    def close(self):
        """
        Closes the file pointer.
        """
        self.fp.close()
