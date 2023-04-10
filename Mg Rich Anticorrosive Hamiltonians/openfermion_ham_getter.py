import numpy as np 

def full_pauli_to_operfermion_pauli(full_pauli):
    """ 
    Convert full pauli representation to openfermion pauli 
    E.g. 'IZIIXX' --->  '[Z1 X4 X5]'
    """
    Sparse_Pauli = '['
    all_I_idx = [i for i, pauli_op in enumerate(full_pauli) if pauli_op == 'I']
    if len(all_I_idx) == len(full_pauli):
        return '[]'
    all_non_I_idx = [i for i in range(len(full_pauli)) if i not in all_I_idx]

    full_pauli_with_I_removal = full_pauli.replace('I',"")
    for idx, pauli_op in zip(all_non_I_idx, full_pauli_with_I_removal):
        Sparse_Pauli += pauli_op 
        Sparse_Pauli += str(idx)
        if idx != all_non_I_idx[-1]:
            Sparse_Pauli += ' '
        elif idx == all_non_I_idx[-1]:
            Sparse_Pauli += ']'
    return Sparse_Pauli 


def full_Hi_to_openfermion_Hi(weight, full_pauli):
    """ 
    return '0.0986007665504536 [Z4 Z14]' 
    """
    openfermion_Hi = ''+str(weight)
    openfermion_Hi += ' '
    openfermion_Hi += full_pauli_to_operfermion_pauli(full_pauli)
    return openfermion_Hi            


def list_weights_and_paulis_to_openfermion_Hamiltonian(weights, paulis):
    """ 
    Convert weight list and paulis list into openfermion Hamiltonian format
    """
    openfermion_ham = []
    for w,p in zip(weights, paulis):
        openfermion_ham.append(full_Hi_to_openfermion_Hi(w,p))
    return openfermion_ham 



