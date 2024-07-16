import openfermion as of
import cirq
from openfermion.circuits import trotter
from openfermion.transforms import get_majorana_operator
import numpy as np

def pow_CCZ(alpha, qubits):
    circuit = cirq.Circuit()
    circuit += cirq.CZPowGate(exponent=alpha/2).on(qubits[1], qubits[2])

    circuit += cirq.CNOT(qubits[0], qubits[1])

    circuit += cirq.CZPowGate(exponent=-alpha/2).on(qubits[1], qubits[2])

    circuit += cirq.CNOT(qubits[0], qubits[1])

    circuit += cirq.CZPowGate(exponent=alpha/2).on(qubits[0], qubits[2])
    return circuit

def replace_ccz_with_custom(circuit):
    """Replace all CCZPowGates in the given circuit with a custom circuit."""
    new_circuit = cirq.Circuit()
    for moment in circuit:
        new_moment = []
        for op in moment.operations:
            if isinstance(op.gate, cirq.CCZPowGate):
                qubits = op.qubits
                alpha = op.gate.exponent
                custom_circuit = pow_CCZ(alpha=alpha, qubits=qubits)
                new_circuit += custom_circuit
            else:
                new_circuit += op
        new_circuit.append(cirq.Moment(new_moment))
    return new_circuit

def compile_to_qasm(circuit, czs=False):
    if czs:
        gateset = cirq.CZTargetGateset(allow_partial_czs=False)
        new_circuit = cirq.optimize_for_target_gateset(circuit, gateset=gateset)
    else:
        new_circuit = circuit
    qasm_str = new_circuit.to_qasm()
    return qasm_str, new_circuit

def count_gates(circuit):
    one_qubit_gates = 0
    two_qubit_gates = 0
    three_qubit_gates = 0

    for moment in circuit:
        for op in moment:
            if len(op.qubits) == 1:
                one_qubit_gates += 1
            elif len(op.qubits) == 2:
                two_qubit_gates += 1
            elif len(op.qubits) == 3:
                three_qubit_gates += 1
                #two_qubit_gates += 5
            else:
                print('large gate detected!')

    return one_qubit_gates, two_qubit_gates, three_qubit_gates


def create_controlled_pauli_string_operator(pauli_tuple):
    # Map the tuple values to Cirq Pauli operators
    pauli_map = {0: cirq.I, 1: cirq.X, 2: cirq.Y, 3: cirq.Z}

    # Create the qubits
    qubits = [cirq.LineQubit(i + 1) for i in range(len(pauli_tuple))]
    control_qubit = cirq.LineQubit(0)

    # Get the Pauli gates for the PauliString
    pauli_gates = [pauli_map[p] for p in pauli_tuple]

    # Create the controlled operation
    controlled_operations = []
    for qubit, gate in zip(qubits, pauli_gates):
        if gate != cirq.I:
            controlled_operations.append(cirq.ControlledGate(gate).on(control_qubit, qubit))

    # Create a circuit to demonstrate the controlled operation
    circuit = cirq.Circuit(controlled_operations)

    return circuit


def sum_amplitudes_by_first_qubit(state_vector):
    sum_first_digit_0 = np.sum(np.abs(state_vector[0::2]) ** 2)
    sum_first_digit_1 = np.sum(np.abs(state_vector[1::2]) ** 2)
    return sum_first_digit_0, sum_first_digit_1


def create_controlled_trotter_circuit(hamiltonian, final_rank, time, n_steps, control_qubit, sys_qubits):
    custom_algorithm = trotter.LowRankTrotterAlgorithm(final_rank=final_rank)
    circuit = cirq.Circuit(
        trotter.simulate_trotter(sys_qubits, hamiltonian,
                                 time=time, omit_final_swaps=True, n_steps=n_steps,
                                 algorithm=custom_algorithm, control_qubit=control_qubit),
        strategy=cirq.InsertStrategy.EARLIEST)
    return circuit


# Define a function to drop iSWAP and T gates
def drop_ts_and_iswap_gates(circuit):
    new_circuit = cirq.Circuit()

    for moment in circuit:
        new_moment = []
        for op in moment:
            if not isinstance(op.gate, cirq.ISwapPowGate) and not isinstance(op.gate, cirq.ZPowGate):
                new_moment.append(op)
        new_circuit.append(cirq.Moment(new_moment))

    return new_circuit


def h_test_circuit(hamiltonian, t1, t2, imag, n_steps, pauli_tuple):
    # Create the controlled trotter circuit
    n_qubits = of.count_qubits(hamiltonian)
    qubits = cirq.LineQubit.range(n_qubits + 1)

    circ = cirq.Circuit(strategy=cirq.circuits.InsertStrategy.EARLIEST)

    # HADAMARD TEST
    circ += cirq.X.on(qubits[2])
    circ += cirq.X.on(qubits[1])

    circ += cirq.H.on(qubits[0])

    CU_t1 = create_controlled_trotter_circuit(hamiltonian,
                                              final_rank=2,
                                              time=t1,
                                              n_steps=n_steps,
                                              control_qubit=qubits[0],
                                              sys_qubits=qubits[1:])
    circ += CU_t1

    CU_t2 = create_controlled_trotter_circuit(hamiltonian,
                                              final_rank=2,
                                              time=t2,
                                              n_steps=n_steps,
                                              control_qubit=qubits[0],
                                              sys_qubits=qubits[1:])

    circ += cirq.X.on(qubits[0])
    circ += CU_t2
    circ += cirq.X.on(qubits[0])

    O_circ = create_controlled_pauli_string_operator(pauli_tuple)

    circ += O_circ

    if imag:
        circ += cirq.S(qubits[0]) ** -1

    # H gate
    circ += cirq.H.on(qubits[0])

    cirq.drop_negligible_operations(circ)

    return circ


def compute_exp(hamiltonian, t1, t2, imag, n_steps, pauli_tuple):
    circ = h_test_circuit(hamiltonian, t1, t2, imag, n_steps, pauli_tuple)
    simulator = cirq.Simulator()
    result = simulator.simulate(circ)

    # Extract the state vector
    final_state_vector = result.final_state_vector

    prob_anc_0, prob_anc_1 = sum_amplitudes_by_first_qubit(final_state_vector)
    expectation_value = prob_anc_0 - prob_anc_1
    return expectation_value
