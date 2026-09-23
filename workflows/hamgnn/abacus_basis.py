# Copyright (c) 2021-2026 HamGNN Team
# SPDX-License-Identifier: GPL-3.0-only

"""ABACUS orbital masks and units used by the HamGNN WFC converter.

Extracted from the HamGNN ABACUS interface utilities. Masks must match the
orbital files and graph generator; nao_max alone does not define a new basis.
"""

import numpy as np
from pymatgen.core.periodic_table import Element


basis_def_13_abacus = (lambda s1=[0],s2=[1],p1=[2,3,4],p2=[5,6,7],d1=[8,9,10,11,12]: {
    1 : np.array(s1+s2+p1, dtype=int), # H
    2 : np.array(s1+s2+p1, dtype=int), # He
    5 : np.array(s1+s2+p1+p2+d1, dtype=int), # B
    6 : np.array(s1+s2+p1+p2+d1, dtype=int), # C
    7 : np.array(s1+s2+p1+p2+d1, dtype=int), # N
    8 : np.array(s1+s2+p1+p2+d1, dtype=int), # O
    9 : np.array(s1+s2+p1+p2+d1, dtype=int), # F
    10: np.array(s1+s2+p1+p2+d1, dtype=int), # Ne
    14: np.array(s1+s2+p1+p2+d1, dtype=int), # Si
    15: np.array(s1+s2+p1+p2+d1, dtype=int), # P
    16: np.array(s1+s2+p1+p2+d1, dtype=int), # S
    17: np.array(s1+s2+p1+p2+d1, dtype=int), # Cl
    18: np.array(s1+s2+p1+p2+d1, dtype=int), # Ar
})()

basis_def_15_abacus = (lambda s1=[0],s2=[1],s3=[2],s4=[3],p1=[4,5,6],p2=[7,8,9],d1=[10,11,12,13,14]: {
    1 : np.array(s1+s2+p1, dtype=int), # H
    2 : np.array(s1+s2+p1, dtype=int), # He
    3 : np.array(s1+s2+s3+s4+p1, dtype=int), # Li
    4 : np.array(s1+s2+s3+s4+p1, dtype=int), # Bi
    5 : np.array(s1+s2+p1+p2+d1, dtype=int), # B
    6 : np.array(s1+s2+p1+p2+d1, dtype=int), # C
    7 : np.array(s1+s2+p1+p2+d1, dtype=int), # N
    8 : np.array(s1+s2+p1+p2+d1, dtype=int), # O
    9 : np.array(s1+s2+p1+p2+d1, dtype=int), # F
    10: np.array(s1+s2+p1+p2+d1, dtype=int), # Ne
    11: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Na
    12: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Mg
    # 13: Al
    14: np.array(s1+s2+p1+p2+d1, dtype=int), # Si
    15: np.array(s1+s2+p1+p2+d1, dtype=int), # P
    16: np.array(s1+s2+p1+p2+d1, dtype=int), # S
    17: np.array(s1+s2+p1+p2+d1, dtype=int), # Cl
    18: np.array(s1+s2+p1+p2+d1, dtype=int), # Ar
})()

basis_def_27_abacus = (lambda s1=[0],s2=[1],s3=[2],s4=[3],p1=[4,5,6],p2=[7,8,9],d1=[10,11,12,13,14],d2=[15,16,17,18,19],f1=[20,21,22,23,24,25,26]: {
    1 : np.array(s1+s2+p1, dtype=int), # H
    2 : np.array(s1+s2+p1, dtype=int), # He
    3 : np.array(s1+s2+s3+s4+p1, dtype=int), # Li
    4 : np.array(s1+s2+s3+s4+p1, dtype=int), # Bi
    5 : np.array(s1+s2+p1+p2+d1, dtype=int), # B
    6 : np.array(s1+s2+p1+p2+d1, dtype=int), # C
    7 : np.array(s1+s2+p1+p2+d1, dtype=int), # N
    8 : np.array(s1+s2+p1+p2+d1, dtype=int), # O
    9 : np.array(s1+s2+p1+p2+d1, dtype=int), # F
    10: np.array(s1+s2+p1+p2+d1, dtype=int), # Ne
    11: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Na
    12: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Mg
    # 13: Al
    14: np.array(s1+s2+p1+p2+d1, dtype=int), # Si
    15: np.array(s1+s2+p1+p2+d1, dtype=int), # P
    16: np.array(s1+s2+p1+p2+d1, dtype=int), # S
    17: np.array(s1+s2+p1+p2+d1, dtype=int), # Cl
    18: np.array(s1+s2+p1+p2+d1, dtype=int), # Ar
    19: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # K
    20: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Ca
    21: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Sc
    22: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Ti
    23: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # V
    24: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Cr
    25: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Mn
    26: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Fe
    27: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Co
    28: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Ni
    29: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Cu
    30: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Zn
    31: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Ga
    32: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Ge
    33: np.array(s1+s2+p1+p2+d1, dtype=int), # As
    34: np.array(s1+s2+p1+p2+d1, dtype=int), # Se
    35: np.array(s1+s2+p1+p2+d1, dtype=int), # Br
    36: np.array(s1+s2+p1+p2+d1, dtype=int), # Kr
    37: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Rb
    38: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Sr
    39: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Y
    40: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Zr
    41: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Nb
    42: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Mo
    43: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Tc
    44: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Ru
    45: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Rh
    46: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Pd
    47: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Ag
    48: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Cd
    49: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # In
    50: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Sn
    51: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Sb
    52: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Te
    53: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # I
    54: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Xe
    55: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Cs
    56: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Ba
    #
    79: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Au
    80: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Hg
    81: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Tl
    82: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Pb
    83: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Bi
})()

basis_def_40_abacus = (lambda s1=[0],
                       s2=[1],
                       s3=[2],
                       s4=[3],
                       p1=[4,5,6],
                       p2=[7,8,9],
                       p3=[10,11,12],
                       p4=[13,14,15],
                       d1=[16,17,18,19,20],
                       d2=[21,22,23,24,25],
                       f1=[26,27,28,29,30,31,32],
                       f2=[33,34,35,36,37,38,39]: {
    Element('Ag').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Al').Z: np.array(s1+s2+s3+s4+p1+p2+p3+p4+d1, dtype=int),
    Element('Ar').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('As').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Au').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Ba').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Be').Z: np.array(s1+s2+s3+s4+p1, dtype=int),
    Element('B').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Bi').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Br').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Ca').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Cd').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('C').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Cl').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Co').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Cr').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Cs').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Cu').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Fe').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('F').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Ga').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Ge').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('He').Z: np.array(s1+s2+p1, dtype=int),
    Element('Hf').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1+f2, dtype=int), # Hf_gga_10au_100Ry_4s2p2d2f.orb
    Element('H').Z: np.array(s1+s2+p1, dtype=int),
    Element('Hg').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('I').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('In').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Ir').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('K').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Kr').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Li').Z: np.array(s1+s2+s3+s4+p1, dtype=int),
    Element('Mg').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Mn').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Mo').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Na').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Nb').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Ne').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('N').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Ni').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('O').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Os').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Pb').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Pd').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('P').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Pt').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Rb').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Re').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Rh').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Ru').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Sb').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Sc').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Se').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('S').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Si').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Sn').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Sr').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Ta').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1+f2, dtype=int), # Ta_gga_10au_100Ry_4s2p2d2f.orb
    Element('Tc').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Te').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Ti').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Tl').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('V').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('W').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1+f2, dtype=int), # W_gga_10au_100Ry_4s2p2d2f.orb
    Element('Xe').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Y').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Zn').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Zr').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int)
    })()

au2ang = 0.5291772490000065

au2ev = 27.211324570273
