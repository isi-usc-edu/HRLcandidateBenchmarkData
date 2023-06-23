import pyscf.tools
import glob
from pyscf import scf, gto, dft, mp, ao2mo, cc, mp, ci, fci
import json
from pyscf import gto,scf,mcscf, fci,lo
from pyscf.scf import ROHF, UHF, ROKS
import numpy as np
import pandas as pd
from pyscf.tools import fcidump
from pyscf.lib import chkfile
from numba import njit

I, X, Y, Z = 0, 1, 2, 3

@njit
def jordan_wigner_transform_4idx(ne, no, hpq, eri):
    n_qubits = no
    terms = []
    coeffs = []

    # Add one-body terms
    for p in range(n_qubits):
        for q in range(n_qubits):
            coeff = hpq[p, q]
            if abs(coeff) > 1e-8:  # Include the integral if it is non-zero
                term = np.full(n_qubits, I)
                term[:p] = Z
                term[p] = X
                term[p+1:q] = Z
                term[q] = Y
                terms.append(term)
                coeffs.append(coeff)

    # Add two-body terms
    for p in range(n_qubits):
        for q in range(n_qubits):
            for r in range(n_qubits):
                for s in range(n_qubits):
                    coeff = eri[p, q, r, s]
                    if abs(coeff) > 1e-8:  # Include the integral if it is non-zero
                        term = np.full(n_qubits, I)
                        term[:p] = Z
                        term[p] = X
                        term[p+1:q] = Z
                        term[q] = X
                        term[q+1:r] = Z
                        term[r] = Y
                        term[r+1:s] = Z
                        term[s] = Y
                        terms.append(term)
                        coeffs.append(coeff)

    return terms, coeffs


   
@njit
def jordan_wigner_transform(hpq, eri_2idx, n_qubits):
    terms = []
    coeffs = []

    # Add one-body terms
    for p in range(n_qubits):
        for q in range(n_qubits):
            coeff = hpq[p, q]
            if coeff > 1e-8:
                term = np.full(n_qubits, I)
                term[:p] = Z
                term[p] = X
                term[p+1:q] = Z
                term[q] = Y
                terms.append(term)
                coeffs.append(coeff)

    # Add two-body terms
    for pq in range(n_qubits**2):
        for rs in range(n_qubits**2):
            coeff = eri_2idx[pq, rs]
            
            if coeff > 1e-8:
                p = pq // n_qubits
                q = pq % n_qubits
                r = rs // n_qubits
                s = rs % n_qubits

                term = np.full(n_qubits, I)
                term[:p] = Z
                term[p] = X
                term[p+1:q] = Z
                term[q] = X
                term[q+1:r] = Z
                term[r] = Y
                term[r+1:s] = Z
                term[s] = Y
                print(coeff)
                terms.append(term)
                coeffs.append(coeff)

    return terms, coeffs

@njit
def AS_jordan_wigner_transform(hpq, eri_2idx, start_orb, end_orb):
    n_qubits = end_orb - start_orb
    terms = []
    coeffs = []

    # Add one-body terms
    for p in range(start_orb, end_orb):
        for q in range(start_orb, end_orb):
            coeff = hpq[p, q]
            if coeff > 1e-8:
                term = np.full(n_qubits, I)
                term[:p] = Z
                term[p - start_orb] = X
                term[p+1 - start_orb:q - start_orb] = Z
                term[q - start_orb] = Y
                terms.append(term)
                coeffs.append(coeff)

    # Add two-body terms
    for pq in range(start_orb**2, end_orb**2):
        for rs in range(start_orb**2, end_orb**2):
            coeff = eri_2idx[pq, rs]

            if coeff > 1e-8:
                p = (pq // n_qubits) + start_orb
                q = (pq % n_qubits) + start_orb
                r = (rs // n_qubits) + start_orb
                s = (rs % n_qubits) + start_orb

                term = np.full(n_qubits, I)
                term[:p - start_orb] = Z
                term[p - start_orb] = X
                term[p+1 - start_orb:q - start_orb] = Z
                term[q - start_orb] = X
                term[q+1 - start_orb:r - start_orb] = Z
                term[r - start_orb] = Y
                term[r+1 - start_orb:s - start_orb] = Z
                term[s - start_orb] = Y
                terms.append(term)
                coeffs.append(coeff)

    return terms, coeffs

if __name__ == '__main__':
    
    el = "CrO"
    basis = "vqz"
    this_chkfile = f'matts_chk_files/{el}_{basis}.chkfile'
    mol = chkfile.load_mol(this_chkfile)
    mf = scf.ROHF(mol)
    scf_result_dic = chkfile.load(this_chkfile, 'scf')
    mf.__dict__.update(scf_result_dic)

    h1 = mf.get_hcore()
    h2 = ao2mo.kernel(mol, mf.mo_coeff)


    H = jordan_wigner_transform(hpq=h1, eri_2idx=h2)
    print(len(H))