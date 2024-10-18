"""

Example usage from an FCIDUMP file

"""
from json_to_metrics_csv import compute_metrics_to_csv
from faux_ham import *
from pyscf.tools import fcidump
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from closedfermion.Models.Molecular.QuarticDirac import QuarticDirac
from closedfermion.Transformations.fermionic_encodings import fermion_to_qubit_transformation
from closedfermion.Transformations.quartic_dirac_transforms import majorana_operator_from_quartic, double_factorization_from_quartic
from json_to_metrics_csv import truncate_df_eigenvalues


if __name__ == "__main__":
    # filenames = get_chk_filenames("chks/")
    #
    # print(filenames)
    #
    chk_file = 'chks/li2_cc-pVDZ_chkfile.chk'
    mol, mf, hcore, norb, eri_4d = load_chk(chk_file)

    quartic_fermion = QuarticDirac(eri_4d, hcore, norb)

    H_DF = double_factorization_from_quartic(quartic_fermion)
    eigs = truncate_df_eigenvalues(H_DF.eigs)

    gs = H_DF.g_mats[:len(eigs)]
    print(np.shape(eigs))
    print(np.shape(gs))
    #
    # """
    #     Truncate gs
    # """
    # print(np.shape(gs))
    # print(np.shape(one_body))


    # filename = 'test.FCIDUMP'

    # metrics = compute_metrics_to_csv(filename=filename, save=False)
    # print(metrics)

    # Generate some sample 2D data
    np.random.seed(42)
    data1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 300)
    data2 = np.random.multivariate_normal([5, 5], [[1, -0.5], [-0.5, 1]], 300)
    data = np.vstack([data1, data2]).T  # Transpose to get shape (n, d)

    # Set up a grid for evaluation of KDE in 2D space
    x_vals, y_vals = np.mgrid[-3:8:100j, -3:8:100j]  # Creates a grid for x and y
    positions = np.vstack([x_vals.ravel(), y_vals.ravel()])

    # Use Gaussian KDE from scipy for 2D
    kde = stats.gaussian_kde(data, bw_method='scott')  # 'scott' or 'silverman' are good defaults
    density = kde(positions).reshape(x_vals.shape)

    # Plot the result
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(x_vals, y_vals, density, cmap='Blues', levels=20)
    ax.scatter(data[0, :], data[1, :], s=5, color='gray', alpha=0.6)
    plt.title("Gaussian KDE in 2D (scipy.stats)")
    plt.xlim([-3, 8])
    plt.ylim([-3, 8])
    plt.show()




