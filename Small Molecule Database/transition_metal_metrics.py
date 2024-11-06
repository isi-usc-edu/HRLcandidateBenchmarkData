import os
import time
import json
import numpy as np
from pyscf import scf, ao2mo
from pyscf.lib import chkfile
import glob
from closedfermion.Models.Molecular.QuarticDirac import QuarticDirac
from closedfermion.Transformations.fermionic_encodings import fermion_to_qubit_transformation
from closedfermion.Transformations.quartic_dirac_transforms import majorana_operator_from_quartic, \
    double_factorization_from_quartic
from compute_metrics import compute_hypergraph_metrics


import math
from scipy.special import comb

def log_fci_hilbert_space_size_approx(n_orbitals, n_electrons):
    # Approximate half of electrons as alpha and half as beta
    n_alpha = n_electrons // 2
    n_beta = n_electrons - n_alpha  # ensures total electron count is preserved
    # Calculate the binomial coefficient for alpha and beta electrons
    hilbert_alpha = comb(n_orbitals, n_alpha, exact=True)
    hilbert_beta = comb(n_orbitals, n_beta, exact=True)
    # Compute the logarithm of the FCI Hilbert space size
    return math.log(hilbert_alpha) + math.log(hilbert_beta)

def get_chk_filenames(path: str, filter_str: str = None):
    fnames = [f for f in os.listdir(path) \
              if os.path.isfile(os.path.join(path, f))]
    if filter_str:
        fnames = [f for f in fnames if filter_str in f]
    return [os.path.join(path, f) for f in fnames]
    # return {f : scf.ROHF(chkfile.load_mol(os.path.join(path, f))) for f in fnames}


def load_chk(chk_file):
    mol = chkfile.load_mol(chk_file)
    mf = scf.ROHF(mol)
    scf_result_dic = chkfile.load(chk_file, 'scf')
    mf.__dict__.update(scf_result_dic)

    nelec = mol.nelectron
    spin = mol.spin

    # H1 = Tne + Vnn
    hcore = mf.get_hcore()

    N_orbs = mol.nao_nr()

    # H2 = Vee
    eri_4d = ao2mo.restore(1, mol.intor('int2e'), N_orbs)

    return mol, mf, hcore, N_orbs, eri_4d, nelec


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


def extract_last_elements(filename):
    # Find the last occurrence of '/' and '.chkfile'
    start_index = filename.rfind('/') + 1
    end_index = filename.rfind('.chkfile')

    # Extract the substring
    if start_index != -1 and end_index != -1:
        return filename[start_index:end_index]
    return None

def compute_metal_metrics(idx):


    path_name = '../Small Molecule Database/transition_metal_chks/'
    chks = glob.glob(path_name+"*.chkfile")
    fname = chks[idx]

    start_time = time.time()

    mol, _, hcore, norb, eri_4d, nelec = load_chk(fname)


    quartic_fermion = QuarticDirac(eri_4d, hcore, norb)
    majorana_op = majorana_operator_from_quartic(quartic_fermion)
    pauli_op = fermion_to_qubit_transformation(majorana_op, 'Jordan-Wigner')

    pauli_data = pauli_op.data
    vertex_degree_stats, weight_stats, edge_order_stats = compute_hypergraph_metrics(pauli_data)
    metrics_target = {}
    for key, val in list(vertex_degree_stats.items()) + list(weight_stats.items()) + list(
            edge_order_stats.items()):
        metrics_target[key] = val

    metrics_target['number_of_terms'] = len(pauli_data.keys())


    H_DF = double_factorization_from_quartic(quartic_fermion)
    eigs = np.array(truncate_df_eigenvalues(H_DF.eigs))

    metrics_target['df_rank'] = len(eigs)
    metrics_target['df_gap'] = abs(eigs[0]-eigs[1])
    metrics_target['n_elec'] = nelec
    metrics_target['n_orb'] = norb
    metrics_target['log_fcisize'] = log_fci_hilbert_space_size_approx(n_orbitals=norb, n_electrons=nelec)


    end_time = time.time()
    metrics_target['wallclock_time'] = end_time-start_time
    filename = extract_last_elements(fname)
    with open(f"./metal_metrics/{filename}.json", "w") as json_file:
        json.dump(metrics_target, json_file, indent=4)
