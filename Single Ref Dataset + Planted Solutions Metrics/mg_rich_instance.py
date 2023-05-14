import openfermion
import glob
import numpy as np
from compute_metrics import compute_cas_metrics
from openfermion import jordan_wigner, bravyi_kitaev
import json

def idx2(i, j, dim):
    """Map the 2D index (i, j) in the lower triangle to a 1D index."""
    if i < j:
        i, j = j, i
    return i * (i + 1) // 2 + j

def idx4(p, q, r, s, dim):
    """Map the 4D index (p, q, r, s) to a 1D index in the 2-index tensor."""
    return idx2(idx2(p, q, dim), idx2(r, s, dim), dim * (dim + 1) // 2)

def four_to_two_index(V):
    dim = V.shape[0]
    v = np.zeros((dim * (dim + 1) // 2, dim * (dim + 1) // 2))
    for p in range(dim):
        for q in range(dim):
            for r in range(dim):
                for s in range(dim):
                    v[idx2(p, q, dim), idx2(r, s, dim)] = V[p, q, r, s]
    return v


def idx2_inv(k, dim):
    """Map the 1D index in the lower triangle to a 2D index (i, j)."""
    i = int((np.sqrt(1 + 8 * k) - 1) // 2)
    j = k - i * (i + 1) // 2
    return i, j

def two_to_four_index(v):
    dim = int(np.sqrt(2 * v.shape[0]))
    V = np.zeros((dim, dim, dim, dim))
    for p in range(v.shape[0]):
        for q in range(v.shape[1]):
            i, j = idx2_inv(p, dim)
            k, l = idx2_inv(q, dim)
            V[i, j, k, l] = v[p, q]
    return V

if __name__ == '__main__':
    f1 = open('1water_pstrings.json')
    pstring1 = json.load(f1)


    compute_cas_metrics(pstring1, path='1water', name='1water')

    f2 = open('2water_pstrings.json')
    pstring2 = json.load(f2)

    compute_cas_metrics(pstring1, path='2water', name='2water')