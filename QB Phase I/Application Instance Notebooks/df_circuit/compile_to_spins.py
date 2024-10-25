import openfermion as of
from utils import count_gates, replace_ccz_with_custom,  h_test_circuit, compile_to_qasm

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
    t1, t2 = .1, -.2
    imag=False

    circ = replace_ccz_with_custom(h_test_circuit(hamiltonian, t1, t2, imag, n_steps, pauli_tuple))



    print(count_gates(circ))
    print(len(circ))
    #
    # qasm, new_circ = compile_to_qasm(circ, True)
    # print(count_gates(new_circ))
    # print(len(new_circ))
    #
    # print(new_circ.to_text_diagram(transpose=True))