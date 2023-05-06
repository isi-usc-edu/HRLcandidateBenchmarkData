from openfermion import QubitOperator
import random
import numpy as np

def create_pauli_string(n):
    pauli_dict = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    pauli_string = ""

    random_pauli_list = [random.randint(0, 3) for _ in range(n)]

    for i, pauli_idx in enumerate(random_pauli_list):
        if pauli_idx != 0:  # 0 represents the identity, so we can skip it
            pauli_string += f"{pauli_dict[pauli_idx]}{i} "

    return QubitOperator(pauli_string) if pauli_string else QubitOperator("I0", 0)


def generate_random_H(n, m):

    random.seed(10)
    rs = [random.choice([-1, 1]) for _ in range(m)]

    H = rs[0]/np.sqrt(m) * create_pauli_string(n)
    for i in range(1,m):
        Pi = create_pauli_string(n)
        ci = rs[i]/np.sqrt(m)

        H += ci*Pi

    return H