import numpy as np
from openfermion import FermionOperator, hermitian_conjugated, circuits
from openfermion.transforms import get_interaction_operator
from closedfermion.Transformations.quartic_dirac_transforms import double_factorization_from_quartic
from faux_ham import *
from json_to_metrics_csv import truncate_df_eigenvalues
import matplotlib.pyplot as plt

import pandas as pd
import os

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


# Function to generate an n x n rotation matrix
def generate_rotation_matrix(n):
    # Generate a random n x n matrix
    random_matrix = np.random.randn(n, n)

    # Use QR decomposition to obtain an orthogonal matrix Q
    Q, R = np.linalg.qr(random_matrix)

    # Ensure the matrix is a proper rotation matrix (det(Q) = 1)
    if np.linalg.det(Q) < 0:
        Q[:, -1] = -Q[:, -1]  # Fix the determinant by flipping the last column

    return Q


if __name__ == "__main__":
    sites = range(7,16)
    #
    n_values = []
    r_values = []
    for n_sites in sites:
        print(n_sites)
        c_i, w_u, d_ij_u = np.random.random(n_sites), np.random.random(n_sites), np.random.random(
            (n_sites, n_sites, n_sites))
        a = 100
        H = agp_hamiltonian_with_coupling(n_sites, a*c_i, a*w_u, a*d_ij_u, a*np.random.random(), a*np.random.random(),
                                          a*np.random.random())

        Q = generate_rotation_matrix(n_sites)
        H.rotate_basis(rotation_matrix=Q)
        quartic_fermion = QuarticDirac(H.two_body_tensor, H.one_body_tensor, 2*n_sites)


        H_DF = double_factorization_from_quartic(quartic_fermion)
        eigs = np.array(truncate_df_eigenvalues(H_DF.eigs))
        n_values.append(2*n_sites)
        r_values.append(len(eigs))
    print(r_values)


    filenames = get_chk_filenames("chks/")

    chks = []
    for chk_file in filenames:
        if 'VDZ' in chk_file:
            print(chk_file)
            chks.append(chk_file)

    ranks = []
    norbs = []
    for fname in tqdm(chks):
        mol, _, hcore, norb, eri_4d = load_chk(fname)

        if norb <= 30:
            quartic_fermion = QuarticDirac(eri_4d, hcore, norb)

            H_DF = double_factorization_from_quartic(quartic_fermion)
            eigs = np.array(truncate_df_eigenvalues(H_DF.eigs))
            ranks.append(len(eigs))
            norbs.append(norb)

    ranks = np.array(ranks)
    print(max(norbs))


    plt.figure(figsize=(10, 6))
    plt.scatter(norbs, ranks)
    # Add titles and labels
    nsites = range(min(norbs), 30 ,1)
    plt.plot(nsites, nsites, marker='o', linestyle='--', color='r', markersize=8, label='$O(N)$')
    plt.plot(nsites, np.array(nsites)**2, marker='o', linestyle='--', color='g', markersize=8, label='$O(N^2)$')
    plt.plot(n_values, r_values, marker='o', linestyle='-', color='b', markersize=8, label='AGP')


    plt.title('Rank vs Number of Sites', fontsize=18)
    plt.xlabel('Number of Sites (n)', fontsize=14)
    plt.ylabel('Rank (r)', fontsize=14)

    # Customize grid, add a legend, and set a nice style
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Show the plot
    plt.show()