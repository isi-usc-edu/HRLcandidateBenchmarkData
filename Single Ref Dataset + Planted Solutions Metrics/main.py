import numpy as np
import cirq
import glob
import numpy as np
from openfermion import jordan_wigner
import openfermion as of
from scipy.linalg import norm
import json
import random
from compute_metrics import compute_metrics
import pandas as pd
import matplotlib.pyplot as plt
from random_hamiltonians import generate_random_H


def c(p):
    if p % 2 == 0:
        j = p // 2
        aj = of.FermionOperator(((j, 0),), 1.)
        aj_dag = of.FermionOperator(((j, 1),), 1.)
        majorana_op = -1j * (aj - aj_dag)
    else:
        j = (p + 1) // 2
        aj = of.FermionOperator(((j, 0),), 1.)
        aj_dag = of.FermionOperator(((j, 1),), 1.)
        majorana_op = aj + aj_dag

    return majorana_op


def H(h, g):
    # h must be a real antisymmetric matrix
    assert np.linalg.norm(h.T + h) <= 1e-9
    assert np.isreal(h.all()) == True

    # Construct free fermion Hamiltonian H0
    e0 = 0.25 * norm(h, 'nuc')

    H = of.FermionOperator((), e0)

    for p in range(len(h)):
        for q in range(len(h)):
            H += 0.25j * h[p][q] * c(p) * c(q)

    # Construct impurity term
    for x, gx in enumerate(g):
        bin_x = np.binary_repr(x)
        mag = sum([int(digit) for digit in bin_x])

        if mag % 2 == 0:
            c_prod = of.FermionOperator((), 1.)
            for p, digit in enumerate(bin_x):
                if int(digit):
                    c_prod *= c(p)

            H += gx * c_prod

    return H

def pauli_sum_to_qubit_operator(pauli_sum):
    qubit_operator = of.QubitOperator()

    for term in pauli_sum:
        coefficient = term.coefficient

        term_tuple = ""
        for i, (qubit, pauli) in enumerate(term.items()):
            term_tuple += str(pauli) + str(qubit.x) + " "

        qubit_operator += of.QubitOperator(term_tuple, coefficient)

    return qubit_operator


def Hspin(H, clifford_circuit):
    H_jw = jordan_wigner(H)
    pstrings = of.transforms.qubit_operator_to_pauli_sum(H_jw)

    strings = []
    for string in pstrings:
        new_string = string.conjugated_by(clifford_circuit)
        strings.append(new_string)
    psum = cirq.PauliSum.from_pauli_strings(strings)

    return pauli_sum_to_qubit_operator(psum)


def generate_random_clifford_circuit(qubits, depth):
    clifford_gates = [cirq.S, cirq.H, cirq.CNOT]

    circuit = cirq.Circuit()

    for _ in range(depth):
        gate = random.choice(clifford_gates)

        if gate == cirq.CNOT:
            qubit1, qubit2 = random.sample(qubits, 2)
            circuit.append(gate(qubit1, qubit2))
        else:
            qubit = random.choice(qubits)
            circuit.append(gate(qubit))

    return circuit


def generate_samples(n, m, sample_id, num_gates, a=100., b=100.):
    num_qubits = n
    qubits = [cirq.GridQubit(i, 0) for i in range(num_qubits)]

    clifford = generate_random_clifford_circuit(qubits, num_gates)

    n = num_qubits
    U = a*np.random.uniform(low=0, high=1.0, size=(2 * n, 2 * n))
    h = np.tril(U) - np.tril(U, -1).T
    for i in range(2 * n):
        h[i, i] = 0

    g = []
    for x in range(2 ** m):
        bin_x = np.binary_repr(x)
        mag = sum([int(digit) for digit in bin_x])

        if mag % 4 == 0:
            gx = b*np.random.random()
            g.append(gx)

        elif mag % 4 == 2:
            gx = b * 1j * np.random.random()
            g.append(gx)

        else:
            gx = b * (np.random.random() + 1j * np.random.random())
            g.append(gx)

    H_jw = Hspin(H(h, g), clifford)

    compute_metrics(H_jw, sample_id=sample_id)


def generate_dataframe():
    samples1 = glob.glob('free_fermion_stats/*_metrics.json')
    samples2 = glob.glob('random_H/*_metrics.json')
    samples3 = glob.glob('molecules/*/*_metrics.json')
    samples = np.concatenate((np.concatenate((samples1, samples2)), samples3))

    analysis = {}

    index = 0
    for sample in samples:
        with open(sample) as user_file:
            f = user_file.read()
        metrics_dict = json.loads(f)
        analysis[index] = metrics_dict
        index += 1
    keys = metrics_dict.keys()
    df = pd.DataFrame.from_dict(analysis).T
    return df, keys


def plot_metrics(df, axes):
    df.plot(kind="scatter", x=axes[0], y=axes[1], ec='r', fc='none')
    plt.savefig(axes[0] + '_vs_' + axes[1] +'.png')

if __name__ == '__main__':
    print('Program started')


    ns = [i for i in range(4, 10)]
    ms = [i for i in range(2, 4)]



    sample_id = 0
    for n in ns:
        for m in ms:
            if n > m:
                for i in range(20):
             
                    #Take 20 samples for each configuration

                    generate_samples(n, m, sample_id, num_gates=20)

                    #H_jw = generate_random_H(n,m)
                    #compute_metrics(H_jw, sample_id)
                    sample_id += 1
    


    df, keys = generate_dataframe()

    axes = list(keys)

    print(len(df))
    for i in range(len(axes)):
        plot_metrics(df, ["induced_norm", axes[i]])



