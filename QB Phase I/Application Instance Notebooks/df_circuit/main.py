import matplotlib.pyplot as plt
from utils import *

if __name__ == '__main__':
    # Set Hamiltonian parameters for LiH simulation in active space.
    diatomic_bond_length = 1.45
    geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., diatomic_bond_length))]
    basis = 'sto-3g'
    multiplicity = 1
    active_space_start = 1
    active_space_stop = 3

    # Generate and populate instance of MolecularData.
    molecule = of.MolecularData(geometry, basis, multiplicity, description="1.45")
    molecule.load()

    # Get the Hamiltonian in an active space.
    hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=range(active_space_start),
        active_indices=range(active_space_start, active_space_stop))

    n_steps = 2
    pauli_tuple = (1, 2, 3, 2)


    def f_real(t1, t2):
        return compute_exp(hamiltonian, t1, t2, imag=False, n_steps=n_steps, pauli_tuple=pauli_tuple)


    def f_imag(t1, t2):
        return compute_exp(hamiltonian, t1, t2, imag=True, n_steps=n_steps, pauli_tuple=pauli_tuple)


    # Define the range and number of samples for each variable
    t1_start, t1_end, t1_samples = -3, 3, 20
    t2_start, t2_end, t2_samples = -3, 3, 20

    # Create linspace for each variable
    t1_linspace = np.linspace(t1_start, t1_end, t1_samples)
    t2_linspace = np.linspace(t2_start, t2_end, t2_samples)

    # Create meshgrid
    T1, T2 = np.meshgrid(t1_linspace, t2_linspace)

    # Initialize the Z matrices
    Z_f = np.zeros_like(T1)
    Z_g = np.zeros_like(T1)

    # Compute the z values for both functions
    for i in range(t1_samples):
        for j in range(t2_samples):
            Z_f[i, j] = f_real(T1[i, j], T2[i, j])
            Z_g[i, j] = f_imag(T1[i, j], T2[i, j])

    # Plot the heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Heatmap for f(t1, t2)
    cf = axs[0].contourf(T1, T2, Z_f, 20, cmap='viridis')
    fig.colorbar(cf, ax=axs[0], label=r'$\text{Re}[\langle O \rangle (t1, t2)]$')
    axs[0].set_xlabel('t1')
    axs[0].set_ylabel('t2')

    # Heatmap for g(t1, t2)
    cg = axs[1].contourf(T1, T2, Z_g, 20, cmap='plasma')
    fig.colorbar(cg, ax=axs[1], label=r'$\text{Im}[\langle O \rangle (t1, t2)]$')
    axs[1].set_xlabel('t1')
    axs[1].set_ylabel('t2')

    plt.show()
