import numpy as np
from openfermion import FermionOperator, hermitian_conjugated

# Function to create the K_ij operator
def K_ij(i, j, c_i, c_j):
    # Spin-up and spin-down components of K_ij
    K_ij_up = c_i * FermionOperator(f'{2 * i}^ {2 * j}') - c_j * FermionOperator(f'{2 * j}^ {2 * i}')
    K_ij_down = c_i * FermionOperator(f'{2 * i + 1}^ {2 * j + 1}') - c_j * FermionOperator(f'{2 * j + 1}^ {2 * i + 1}')
    return K_ij_up + K_ij_down


# Function to add N, N^2, and Omega operators
def add_n_n2_omega_operators(n_sites):
    hamiltonian = FermionOperator()

    # Add N operator: sum over spin-up and spin-down number operators
    N_operator = FermionOperator()
    for i in range(n_sites):
        N_operator += FermionOperator(f'{2*i}^ {2*i}')  # n_i,up
        N_operator += FermionOperator(f'{2*i+1}^ {2*i+1}')  # n_i,down

    # Add N^2 operator
    N_squared_operator = N_operator * N_operator

    # Add Omega operator
    Omega_operator = FermionOperator()
    for i in range(n_sites):
        n_up = FermionOperator(f'{2*i}^ {2*i}')  # n_i,up
        n_down = FermionOperator(f'{2*i+1}^ {2*i+1}')  # n_i,down
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

    return hamiltonian + alpha * N_operator + beta * N_squared_operator + gamma * Omega_operator

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