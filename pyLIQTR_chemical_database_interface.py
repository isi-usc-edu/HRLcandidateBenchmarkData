import re

import numpy as np
import json
import pyLIQTR.QSP.QSP as pQSP
import pyLIQTR.QSP.gen_qsp as qspFuncs
from pyLIQTR.QSP.Hamiltonian import Hamiltonian as pyH

np.set_printoptions(precision=2)

def convert_pauli_hamiltonian(hamiltonian):
    """
    A function for converting a Pauli string Hamiltonian stored in a dictionary format
    to a nested list format used by the functions for computing Trotter spectral error bounds
    and the graph metrics
    """
    pauli_strings = []
    coefficients = []

    max_index_global = 0

    # Find the max_index_global
    for key in hamiltonian.keys():
        if key == "[]":
            continue

        terms = key.split()
        term_data = [(term[0], int(re.search(r'\d+', term).group())) for term in terms if re.search(r'\d+', term)]
        max_index = max([index for _, index in term_data])
        max_index_global = max(max_index_global, max_index)

    # Process the terms
    for key, value in hamiltonian.items():
        if key == "[]":
            continue

        terms = key.split()
        term_data = [(term[0], int(re.search(r'\d+', term).group())) for term in terms if re.search(r'\d+', term)]

        formatted_key = ""
        for i in range(max_index_global + 1):
            current_term = ''.join([term for term, index in term_data if index == i]) or "I"
            if current_term != '[':  # Ensure the current term is not an identity operator
                formatted_key += current_term

        pauli_strings.append(formatted_key)
        coefficients.append(value)

    if "[]" in hamiltonian:
        identity_length = len(pauli_strings[0])
        pauli_strings.append("I" * identity_length)
        coefficients.append(hamiltonian["[]"])

    return [pauli_strings, coefficients]

if __name__ == '__main__':


    f = open('1.0_h2_CAS_pstring.json')
    p_dict = json.load(f)


    pstrings = convert_pauli_hamiltonian(p_dict)
    ham_strings = []
    for i, pstring in enumerate(pstrings[0]):
        ham_strings.append((pstring, float(pstrings[1][i])))

    qsp_H = pyH(ham_strings)


    required_precision = 1e-2

    angles = []

    angles.append(qspFuncs.compute_hamiltonian_angles(qsp_H, 1, required_precision, mode="random")[0])

    for n in range(len(angles)):
        print(1, "\t", len(angles[n]))

    qsp_generator = pQSP.QSP(phis=angles[0], hamiltonian=qsp_H, target_size=qsp_H.problem_size)
    curr_circ = qsp_generator.circuit()

    print(curr_circ)