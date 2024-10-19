"""

Example usage from an FCIDUMP file

"""
from json_to_metrics_csv import compute_metrics_to_csv
from faux_ham import *
from pyscf.tools import fcidump
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from closedfermion.Models.Molecular.QuarticDirac import QuarticDirac
from closedfermion.Transformations.fermionic_encodings import fermion_to_qubit_transformation
from closedfermion.Transformations.quartic_dirac_transforms import majorana_operator_from_quartic, double_factorization_from_quartic
from json_to_metrics_csv import truncate_df_eigenvalues

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    filenames = get_chk_filenames("chks/")


    chks=[]
    for chk_file in filenames:
        if 'VDZ' in chk_file and 'h5' not in chk_file:
            print(chk_file)
            chks.append(chk_file)




    g_reduced_vectors=[]
    norbs = []

    for fname in tqdm(chks):


        mol, _, hcore, norb, eri_4d = load_chk(fname)

        if norb <= 20:

            quartic_fermion = QuarticDirac(eri_4d, hcore, norb)

            H_DF = double_factorization_from_quartic(quartic_fermion)
            eigs = np.array(truncate_df_eigenvalues(H_DF.eigs))

            gs = H_DF.g_mats[:len(eigs)]
            # Step 1: Flatten all the tensors into a vector of length L * N * N
            g_flattened = gs.reshape(len(eigs), norb * norb)  # Flatten each tensor to a vector of size N*N

            # Step 2: Use the lambdas to compute the weighted sum of flattened tensors
            g_reduced = np.dot(eigs/np.linalg.norm(eigs), g_flattened)  # Weighted sum across the L tensors

            # Step 3: Store the reduced vector (size N * N) for the i-th sample
            g_reduced_vectors.append(g_reduced)

            norbs.append(norb)

    print(max(norbs))
    # # Step 1: Find the maximum length of the vectors
    gs_sample = []
    max_length = max(norbs)**2
    for g in g_reduced_vectors:
        g_vec = np.concatenate((g, np.zeros(max_length-len(g))))
        print(len(g_vec))
        gs_sample.append(g_vec)

    gs_sample = np.array(gs_sample)
    #
    # # Assuming g_reduced_vectors is already computed (size M x N*N)
    M, dim = gs_sample.shape  # M samples and original dimensionality (N*N)
    print(len(gs_sample), dim)
    #
    # Step 1: Dimensionality reduction using PCA
    pca = PCA(n_components=min(M - 1, dim))  # Use M-1 components or fewer to avoid overfitting
    g_reduced_pca = pca.fit_transform(gs_sample)

    # Step 2: Fit a Gaussian Mixture Model (GMM) to the reduced data for sampling
    gmm = GaussianMixture(n_components=2)  # Example with 2 Gaussian components
    gmm.fit(g_reduced_pca)

    # Step 3: Sample from the GMM in the reduced space
    n_samples = 100  # Example: sample 50 new points
    g_samples_reduced = gmm.sample(n_samples=n_samples)[0]

    # Step 4: Optionally, inverse transform the sampled data back to the original space
    g_samples_original = pca.inverse_transform(g_samples_reduced)

    # Step 5: Plot the first two dimensions of the sampled data compared to the original
    g_sampled_two_dims = g_samples_original[:, :2]  # Sampled data (first two dimensions)
    #
    # g_pca_projected = pca.fit_transform(g_reduced_vectors)  # Project the data onto PCA axes

    # Step 3: Plot the original data along the new PCA axes
    plt.figure(figsize=(8, 6))
    plt.title('Original Data Projected onto New PCA Axes')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.scatter(g_sampled_two_dims[:, 0], g_sampled_two_dims[:, 1], color='red', label='Sampled Data', alpha=0.6)
    plt.title('Original vs Sampled Data (First Two Dimensions)')

    plt.legend()
    plt.grid(True)
    plt.show()

