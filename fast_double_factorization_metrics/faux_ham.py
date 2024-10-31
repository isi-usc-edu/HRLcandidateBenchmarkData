#!/usr/bin/env python3

import sys, os
import pandas as pd
import numpy as np
import scipy as sp

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from pyscf import scf, ao2mo, symm, fci, cc, gto
from pyscf.lib import chkfile
from scipy.special import comb

from tqdm import tqdm


# sys.path.append("../closed-fermion-main")
# from RandomDF import uniform_random_DF
# import closedfermion as cf

sys.path.append("")
from closedfermion.Models.Molecular.QuarticDirac import QuarticDirac
from compute_metrics import *

import logging
logger = logging.getLogger(__name__)


def get_chk_filenames(path: str, filter_str: str = None):
    fnames = [f for f in os.listdir(path) \
              if os.path.isfile(os.path.join(path, f))]
    if filter_str:
        fnames = [f for f in fnames if filter_str in f]
    return [os.path.join(path, f) for f in fnames]
    # return {f : scf.ROHF(chkfile.load_mol(os.path.join(path, f))) for f in fnames}


def load_chk(chk_file):
    try:
        mol = chkfile.load_mol(chk_file)
        mf = scf.ROHF(mol)
        scf_result_dic = chkfile.load(chk_file, 'scf')
        mf.__dict__.update(scf_result_dic)
    except OSError as err:
        logger.error(f"Caught OSError ({chk_file}): {err}")
        return None,None,None,None,None
    except KeyError as err:
        logger.error(f"Caught OSError ({chk_file}): {err}")
        return None,None,None,None,None
    
    nelec = mol.nelectron
    spin = mol.spin

    # H1 = Tne + Vnn
    hcore = mf.get_hcore()

    N_orbs = mol.nao_nr()
    
    # H2 = Vee
    eri_4d = ao2mo.restore(1, mol.intor('int2e'), N_orbs)
    
    return mol, mf, hcore, N_orbs, eri_4d


def create_density_estimator(chk_filenames: list[str], ntile: int=1,
                             num_itr: int=1000, verbose=False) \
        -> KernelDensity:
    """

    Arguments:
       chk_filenames (list): Names of chk files.

       ntile (int): the `n`-quantile of the chk files to be included in the
          density estimator in terms of file size. For example, to use only
          the smallest 20% of the chk files set ntile to 5. To use all of
          the chk files set ntile to 1.

       num_itr (int): number of iterations for GridSearch (default=1000).
    """
    assert ntile >= 1, f"Invalid value for ntile ({ntile})."
    assert num_itr >= 1, f"Invalid value for num_itr ({num_itr})."
    
    # Get file names and sort by file size
    chks = sorted(chk_filenames, key=os.path.getsize)

    
    # Get data from chk files
    orbs, nelec, lamb_cnts = list(), list(), list()
    k = int(len(chks) / ntile)

    if verbose: print("Preparing data...")
    for fname in tqdm(chks[:k-1], disable=(not verbose)):
        mol, _, _, N_orbs, eri_4d = load_chk(fname)
        if mol is None: continue
        N_sq = N_orbs ** 2
        W = np.reshape(eri_4d, (N_sq, N_sq))
        l, _ = np.linalg.eigh(W)
        orbs.append(N_orbs)
        nelec.append(mol.nelectron)
        lamb_cnts.append(l.shape[0])

        
    # Prepare and fit data
    if verbose: print("Fitting density estimator...")
    data = list(zip(orbs, nelec, lamb_cnts))
    params = {"bandwidth" : np.logspace(-1,1,num_itr) / 10}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(data)
    kde = grid.best_estimator_
    if verbose: print("Done.")

    return kde # return estimator


def construct_singlet_excitations_unitary(n, thetas):
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            m[i, j] = thetas[i, j]
            m[j, i] = -m[i, j]

    return sp.linalg.expm(m)


def construct_df_spatial_orbital(N, M, alphas_1darray_m, epsilon_2darray_mp, thetas_3darray_mpq):
    # two_body_tensor assumes the form: H = \sum_{ijkl} g_{ijkl} a_i^\dagger a_j a_k^\dagger a_l
    two_body_tensor = np.zeros((N, N, N, N))
    for miter in range(M):
        alpha = alphas_1darray_m[miter]
        epsilon_1darray_p = epsilon_2darray_mp[miter, :]
        epsilon_2darray_pq = np.einsum("p,q->pq", epsilon_1darray_p, epsilon_1darray_p)
        epsilon_4darray_ppqq = np.zeros((N, N, N, N))
        for i in range(N):
            for j in range(N):
                epsilon_4darray_ppqq[i, i, j, j] = epsilon_2darray_pq[i, j]

        unitary_rotation_pq = construct_singlet_excitations_unitary(N, thetas_3darray_mpq[miter, :, :])
        two_body_tensor += alpha * np.einsum("ppqq,pi,pj,qk,ql->ijkl", epsilon_4darray_ppqq, unitary_rotation_pq,
                                             unitary_rotation_pq, unitary_rotation_pq, unitary_rotation_pq)

    return two_body_tensor


def convert_to_physicist(chemist_tbt):
    # original tbt assumes the form: H = \sum_{ijkl} g_{ijkl} a_i^\dagger a_j a_k^\dagger a_l
    # phys_obt and phys_tbt assume the form:
    #   H = \sum_{pq} H_pq a_p^\dagger a_q + 0.5*\sum_{pqrs} G_{pqrs} a_p^\dagger a_r^\dagger a_s a_q
    # H_pq = \sum_{k} g_{pkkq}
    # This is the form the fcidump file is in
    phys_obt = np.einsum("ikkj->ij", chemist_tbt)
    phys_tbt = 2 * chemist_tbt
    return phys_obt, phys_tbt


def check_permutation_symmetries_real_orbitals(one_body_tensor, two_body_tensor):
    # Works for both spin-orbital and orbital tensors
    # The symmetries tested here are valid only if the underlying ORBITALS
    # (or SPIN-ORBITALS) are real,
    # NOT if the one_body_tensor and two_body_tensor elements are real.
    # The orbitals can be complex, while the elements of the one_body_tensor and
    # two_body_tensor could still be real.

    symm_check_passed = True
    num_orbitals = one_body_tensor.shape[0]
    for p in range(num_orbitals):
        for q in range(num_orbitals):
            if not np.allclose(one_body_tensor[p, q], one_body_tensor[q, p]):
                symm_check_passed = False
                print.warning(
                    f"one_body_tensor[{p}, {q}] != one_body_tensor[{q}, {p}]: {one_body_tensor[p, q]} != {one_body_tensor[q, p]}"
                )

            for r in range(num_orbitals):
                for s in range(num_orbitals):
                    if (
                            not np.allclose(
                                two_body_tensor[p, q, r, s], two_body_tensor[r, s, p, q]
                            )
                            or not np.allclose(
                        two_body_tensor[p, q, r, s], two_body_tensor[q, p, r, s]
                    )
                            or not np.allclose(
                        two_body_tensor[p, q, r, s], two_body_tensor[p, q, s, r]
                    )
                            or not np.allclose(
                        two_body_tensor[p, q, r, s], two_body_tensor[q, p, s, r]
                    )
                            or not np.allclose(
                        two_body_tensor[p, q, r, s], two_body_tensor[r, s, q, p]
                    )
                            or not np.allclose(
                        two_body_tensor[p, q, r, s], two_body_tensor[s, r, p, q]
                    )
                            or not np.allclose(
                        two_body_tensor[p, q, r, s], two_body_tensor[s, r, q, p]
                    )
                    ):
                        symm_check_passed = False

                        print(
                            f"Permutation check of two body tensor failed.\n"
                            + f"two_body_tensor[{p},{q},{r},{s}] = {two_body_tensor[p, q, r, s]}\n"
                            + f"two_body_tensor[{r},{s},{p},{q}] = {two_body_tensor[r, s, p, q]}\n"
                            + f"two_body_tensor[{q},{p},{r},{s}] = {two_body_tensor[q, p, r, s]}\n"
                            + f"two_body_tensor[{p},{q},{s},{r}] = {two_body_tensor[p, q, s, r]}\n"
                            + f"two_body_tensor[{q},{p},{s},{r}] = {two_body_tensor[q, p, s, r]}\n"
                            + f"two_body_tensor[{r},{s},{q},{p}] = {two_body_tensor[r, s, q, p]}\n"
                            + f"two_body_tensor[{s},{r},{p},{q}] = {two_body_tensor[s, r, p, q]}\n"
                            + f"two_body_tensor[{s},{r},{q},{p}] = {two_body_tensor[s, r, q, p]}\n"
                        )
    return symm_check_passed


def truncate_df_eigenvalues(lambs, threshold=1e-8):
    """
    Truncate all values in the list below the specified threshold.

    Args:
    values (list of float): The list of values to truncate.
    threshold (float): The threshold below which values will be truncated.

    Returns:
    list of float: The list with values below the threshold truncated to zero.
    """
    new_list = []
    for value in lambs:
        if value >= threshold:
            new_list.append(value)
    return np.sort(new_list)




def faux_mol(norb, nelec, neigs):
    """
    
    Parameters:
       - norb (int): number of orbitals.

       - nelec (int): number of electrons, max 2 * norb.

       - neigs (int): number of eigen values, max norb**2 but typically O(norb).
    """
    
    alphas_1darray_m = np.random.rand(neigs)
    epsilon_2darray_mp = np.random.rand(neigs, norb)
    thetas_3darray_mpq = np.random.rand(neigs, norb, norb)
    for i in range(neigs):
        thetas_3darray_mpq[i] = np.tril(thetas_3darray_mpq[i], -1)
        
    two_body_tensor = \
        construct_df_spatial_orbital(norb, neigs,
                                     alphas_1darray_m,
                                     epsilon_2darray_mp,
                                     thetas_3darray_mpq)
        
    h1, h2 = convert_to_physicist(two_body_tensor)
    
    mol = gto.M(verbose=0)
    mol.nelectron = nelec
    hf = scf.RHF(mol)
    hf._eri = ao2mo.restore(8, h2, norb)
    hf.get_hcore = lambda *args: h1
    hf.get_ovlp = lambda *args: np.eye(norb)
    hf.kernel()

    mol_nelec = mol.nelectron
    spin = mol.spin
    hcore = hf.get_hcore()
    N_orbs = mol.nao_nr()
    eri_4d = ao2mo.restore(1, mol.intor('int2e'), N_orbs)

    mycc = cc.CCSD(hf)
    eris = mycc.ao2mo()
    e_corr, t1, t2 = mycc.kernel(eris=eris)
    # Now add the (T) correction to get the CCSD(T) energy
    e_ccsdt = mycc.ccsd_t()

    # Total energy is the sum of SCF energy, CCSD correlation energy, and (T) correction
    e_tot_ccsdt = mycc.e_tot + e_ccsdt

    # Initialize the FCI solver
    cisolver = fci.FCI(hf)

    # Compute the FCI energy and wavefunction
    e_fci, ci = cisolver.kernel()
    
    # The total FCI energy is just the computed energy since FCI is exact within the given space
    # print("FCI Total Energy:", abs(e_fci-e_tot_ccsdt))

    return {'hf' : hf,
            'norb' : N_orbs,
            'nelec' : mol_nelec,
            'hcore' : hcore,
            'e_ccsdt' : e_ccsdt,
            'e_tot_ccsdt' : mycc.e_tot + e_ccsdt,
            'e_fci' : e_fci,
            'ci' : ci,
            'fci_total_energy' : abs(e_fci - e_tot_ccsdt)}


def get_mol_metrics(fm):
    """Returns a dict of metrics for a given molecule.

    Parameters:
       fm (dict): A fake molecule as returned by `faux_mol`.
    """
    mets = dict()
    hf = fm['hf']
    mol = hf.mol

    hcore = hf.get_hcore()
    nelec = mol.nelectron
    N_orbs = mol.nao_nr()
    eri_4d = ao2mo.restore(1, mol.intor('int2e'), N_orbs)

    QF = QuarticDirac(V=eri_4d, h=hcore, N_orbs=N_orbs)
    one_body, lambs, g_mats = QF.double_factorization(purely_quartic=False)

    MO = QF.to_majorana_operator(enumerative=True)

    MO._clear_zero_terms()

    JW_Pauli_op, one_norm, pauli_terms = MO.jordan_wigner_transform()

    vertex_degree_stats, weight_stats, edge_order_stats = compute_hypergraph_metrics(JW_Pauli_op)

    metrics_list = list(vertex_degree_stats.items()) + list(weight_stats.items()) + list(edge_order_stats.items())
    metrics = {k : v for (k, v) in metrics_list}
        
    metrics['one_norm'] = one_norm
    metrics['number_of_terms'] = pauli_terms
    metrics['n_elec'] = nelec
    metrics['n_orbs'] = N_orbs
    metrics['e_ccsdt'] = fm['e_ccsdt']
    metrics['e_tot_ccsdt'] = fm['e_tot_ccsdt']
    metrics['e_fci'] = fm['e_fci']
    metrics['ci'] = fm['ci']
    metrics['fci_total_energy'] = fm['fci_total_energy']
    
    return metrics
