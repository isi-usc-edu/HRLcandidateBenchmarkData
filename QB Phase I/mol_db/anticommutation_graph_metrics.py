from collections import defaultdict
import re
import hypernetx as hnx
import numpy as np
import itertools
import networkx as nx

def pauli_string_to_hyperedge(pauli_string):
    """
        Hyperedges of a Pauli string are of the form

        (ik_1, ik_2)

        where the associated Pauli string is P_i = σ_i1 x ... x σ_in
        and indices ik_j are those such that σ_ik_j and σ_ik_(j+1) do not commute

        Args:
            str: Pauli string "XIXZIIY"
        Returns:
            tuple: (ik_1, ik_2)
    """
    pauli_operators = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    anticommuting_pairs = set()

    for i in range(len(pauli_string)):
        op1 = pauli_string[i]
        for j in range(i + 1, len(pauli_string)):
            op2 = pauli_string[j]
            if op1 != 'I' and op2 != 'I' and pauli_operators[op1] ^ pauli_operators[op2] == 3:
                anticommuting_pairs.add((i, j))

    return tuple(anticommuting_pairs)


def build_hypergraph(pauli_strings):
    """
        Construct Anticommuting Pauli Hypergraph from a list of Pauli strings
        Args:
            list: Pauli strings ["XIXZIIY", "IIYYYZZ", ...]

        Returns:
            set: Anticommuting Pauli Hypergraph G = (E,V) for E = {(ik_1, ik_2)}, V = {1, 2, ..., n} for n qubits
    """
    hypergraph = defaultdict(set)

    for pauli_string in pauli_strings:
        hyperedges = pauli_string_to_hyperedge(pauli_string)
        for hyperedge in hyperedges:
            for vertex in hyperedge:
                hypergraph[vertex].add(hyperedge)

    return hypergraph
