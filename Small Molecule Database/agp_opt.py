import numpy as np
from openfermion import FermionOperator, hermitian_conjugated, circuits
from openfermion.transforms import get_interaction_operator
from closedfermion.Transformations.quartic_dirac_transforms import double_factorization_from_quartic
from faux_ham import *
from json_to_metrics_csv import truncate_df_eigenvalues
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from closedfermion.Transformations.fermionic_encodings import fermion_to_qubit_transformation
from closedfermion.Transformations.quartic_dirac_transforms import majorana_operator_from_quartic, double_factorization_from_quartic
import numpy as np
from scipy.optimize import minimize

import numpy as np
from scipy.linalg import expm


def rotation_matrix(n, parameters):
    """
    Generate a rotation matrix for n orbitals by exponentiating an antisymmetric matrix.

    Parameters:
        n (int): Dimension of the orbital space (nxn matrix).
        parameters (list or array): List of n*(n-1)/2 free parameters for the antisymmetric matrix.

    Returns:
        R (ndarray): The resulting rotation matrix.
    """
    # Ensure the correct number of parameters is provided
    expected_num_params = n * (n - 1) // 2
    if len(parameters) != expected_num_params:
        raise ValueError(f"Expected {expected_num_params} parameters, but got {len(parameters)}.")

    # Initialize an nxn matrix with zeros
    A = np.zeros((n, n))

    # Fill the upper triangle of the matrix with parameters, making it antisymmetric
    param_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = parameters[param_idx]
            A[j, i] = -parameters[param_idx]  # Ensuring antisymmetry
            param_idx += 1

    # Exponentiate the antisymmetric matrix to get the rotation matrix
    R = expm(A)
    return R

# Function to create the K_ij operator
def K_ij(i, j, c_i, c_j):
    # Spin-up and spin-down components of K_ij
    K_ij_up = c_i * FermionOperator(f'{2 * i}^ {2 * j}') - c_j * FermionOperator(f'{2 * j}^ {2 * i}')
    K_ij_down = c_i * FermionOperator(f'{2 * i + 1}^ {2 * j + 1}') - c_j * FermionOperator(f'{2 * j + 1}^ {2 * i + 1}')
    return K_ij_up + K_ij_down


# Function to add N, N^2, and Omega operators
def add_n_n2_omega_operators(n_sites):
    N_operator = FermionOperator()
    for i in range(n_sites):
        N_operator += FermionOperator(f'{2 * i}^ {2 * i}')  # n_i,up
        N_operator += FermionOperator(f'{2 * i + 1}^ {2 * i + 1}')  # n_i,down

    # Add N^2 operator
    N_squared_operator = N_operator * N_operator

    # Add Omega operator
    Omega_operator = FermionOperator()
    for i in range(n_sites):
        n_up = FermionOperator(f'{2 * i}^ {2 * i}')  # n_i,up
        n_down = FermionOperator(f'{2 * i + 1}^ {2 * i + 1}')  # n_i,down
        Omega_operator += n_up + n_down - 2 * n_up * n_down

    return N_operator, N_squared_operator, Omega_operator


# Create the AGP Hamiltonian
def agp_hamiltonian_with_coupling(n_sites, c_i, w_u, d_ij_u, alpha=0.0, beta=0.0, gamma=0.0):
    hamiltonian = FermionOperator()

    # Double sum over i > j and k > l
    for i in range(n_sites):
        for j in range(i):
            Kij = K_ij(i, j, c_i[i], c_i[j])  # Compute K_ij

            for k in range(n_sites):
                for l in range(k):
                    Kkl = K_ij(k, l, c_i[k], c_i[l])  # Compute K_kl

                    # Sum over u terms
                    coupling_sum = 0
                    for u in range(len(w_u)):
                        coupling_sum += w_u[u] * d_ij_u[u, i, j] * d_ij_u[u, k, l]

                    # Add the K_ij^\dagger K_kl terms to the Hamiltonian
                    hamiltonian += coupling_sum * hermitian_conjugated(Kij) * Kkl

    N_operator, N_squared_operator, Omega_operator = add_n_n2_omega_operators(n_sites)

    fermion_operator = hamiltonian + alpha * N_operator + beta * N_squared_operator + gamma * Omega_operator

    return get_interaction_operator(fermion_operator)



def get_agp_e0(alpha, beta, Np):
    return alpha * Np + beta * Np**2



# Define the cost function that accepts a flattened vector x
def cost_function(x, rank, n_sites, target, get_metrics=False):
    # Calculate the split indices based on the shapes of c_i, w_u, and d_ij_u
    a, b, c = x[0], x[1], x[2]

    # Calculate the split indices for the remaining parameters
    c_i_size = n_sites
    w_u_size = rank
    d_ij_u_size = rank * n_sites * n_sites

    # Reshape the rest of x into the respective parameters
    c_i = x[3:3 + c_i_size]
    w_u = x[3 + c_i_size:3 + c_i_size + w_u_size]
    d_ij_u = x[3 + c_i_size + w_u_size:3 + c_i_size + w_u_size + d_ij_u_size].reshape(
        (rank, n_sites, n_sites))
    rot_params = x[3 + c_i_size + w_u_size + d_ij_u_size:]  # Extract the new array


    H = agp_hamiltonian_with_coupling(n_sites, c_i, w_u, d_ij_u, a, b ,c)

    U = rotation_matrix(n_sites, rot_params)
    H.rotate_basis(rotation_matrix=U)
    quartic_fermion = QuarticDirac(H.two_body_tensor, H.one_body_tensor, 2 * n_sites)

    H_DF = double_factorization_from_quartic(quartic_fermion)

    if get_metrics == True:
        majorana_op = majorana_operator_from_quartic(quartic_fermion)
        pauli_op = fermion_to_qubit_transformation(majorana_op, 'Jordan-Wigner')

        pauli_data = pauli_op.data
        vertex_degree_stats, weight_stats, edge_order_stats = compute_hypergraph_metrics(pauli_data)
        metrics_target = {}
        for key, val in list(vertex_degree_stats.items()) + list(weight_stats.items()) + list(
                edge_order_stats.items()):
            metrics_target[key] = val

        metrics_target['number_of_terms'] = len(pauli_data.keys())
        return metrics_target, np.array(H_DF.eigs)

    eigs = np.array(H_DF.eigs)


    return np.linalg.norm(eigs - target)


if __name__ == "__main__":

    fname = 'libe.chk'

    mol, _, hcore, norb, eri_4d = load_chk(fname)



    quartic_fermion = QuarticDirac(eri_4d, hcore, norb)

    H_DF = double_factorization_from_quartic(quartic_fermion)


    eigs = np.array(H_DF.eigs)
    # gs = H_DF.g_mats
    # target = np.concatenate([eigs, gs.flatten()])


    rank = 2
    n_sites = norb//2

    def f(x):
        cost = cost_function(x, rank, n_sites, eigs)
        print(cost)
        return cost


    # Initial guess for x, with a, b, c at the front
    initial_a, initial_b, initial_c = np.random.random(3)
    initial_c_i = np.random.random(n_sites)
    initial_w_u = np.random.random(rank)
    initial_d_ij_u = np.random.random((rank, n_sites, n_sites))
    initial_rot = np.random.random(n_sites*(n_sites-1)//2)
    initial_x = np.concatenate([[initial_a, initial_b, initial_c], initial_c_i, initial_w_u, initial_d_ij_u.flatten(), initial_rot])

    # Optimize using SciPy's minimize function
    result = minimize(f, initial_x, method='BFGS')  # Use BFGS; can also try 'Nelder-Mead' or other methods

    # Check optimization results
    optimized_x = result.x
    optimized_cost = result.fun

    majorana_op = majorana_operator_from_quartic(quartic_fermion)
    pauli_op = fermion_to_qubit_transformation(majorana_op, 'Jordan-Wigner')

    pauli_data = pauli_op.data
    vertex_degree_stats, weight_stats, edge_order_stats = compute_hypergraph_metrics(pauli_data)
    metrics_target = {}
    for key, val in list(vertex_degree_stats.items()) + list(weight_stats.items()) + list(
            edge_order_stats.items()):
        metrics_target[key] = val

    metrics_target['number_of_terms'] = len(pauli_data.keys())

    metric_list, eigs_list = cost_function(optimized_x, rank, n_sites, eigs, get_metrics=True)
    print(metrics_target)
    print(metric_list)
    print(len(truncate_df_eigenvalues(eigs_list)))
    print(len(truncate_df_eigenvalues(eigs)))

