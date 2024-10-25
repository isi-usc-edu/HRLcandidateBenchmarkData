from collections import defaultdict
import re
import numpy as np


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


def pauli_string_to_hyperedge(pauli_string):
    """
        Hyperedges of a Pauli string are of the form

        (ik_1, ik_2, ..., ik_m)

        where the associated Pauli string is P_i = σ_i1 x ... x σ_in
        and indices ik_j are those such that σ_ik_j != 1

        Args:
            str: Pauli string "XIXZIIY"
        Returns:
            tuple: (ik_1, ik_2, ..., ik_m)
    """
    return tuple(i for i, op in enumerate(pauli_string) if op != 'I')


def build_hypergraph(pauli_strings):
    """
        Construct Pauli Hypergraph from a list of Pauli strings
        Args:
            list: Pauli strings ["XIXZIIY", "IIYYYZZ", ...]

        Returns:
            set: Pauli Hypergraph G = (E,V) for E = {(ik_1, ik_2, ..., ik_m)}, V = {1, 2, ..., n} for n qubits
    """
    hypergraph = defaultdict(set)

    for pauli_string in pauli_strings:
        hyperedge = pauli_string_to_hyperedge(pauli_string)
        for vertex in hyperedge:
            hypergraph[vertex].add(hyperedge)

    return hypergraph


def vertex_degree_stats(hypergraph):
    """
    Compute the number of hyperedges each vertex is a member of, and then
    compute the max and average of that set of numbers.

    Args:
    hypergraph (dict): A dictionary representing a hypergraph, where keys are vertices
                       and values are sets of hyperedges.

    Returns:
    tuple: A tuple containing the max and average of the number of hyperedges each vertex
           is a member of.
    """
    num_hyperedges_per_vertex = []

    for vertex in range(len(hypergraph)):
        num_hyperedges_per_vertex.append(len(hypergraph[vertex]))

    max_hyperedges = max(num_hyperedges_per_vertex)
    min_hyperedges = min(num_hyperedges_per_vertex)
    avg_hyperedges = sum(num_hyperedges_per_vertex) / len(num_hyperedges_per_vertex)

    # Compute the standard deviation
    variance = sum((deg - avg_hyperedges) ** 2 for deg in num_hyperedges_per_vertex) / len(num_hyperedges_per_vertex)
    std_dev = np.sqrt(variance)

    return num_hyperedges_per_vertex, max_hyperedges, min_hyperedges, avg_hyperedges, std_dev


def hyperedge_orders_stats(hypergraph):
    """
    Compute the max, average, minimum, and standard deviation of the set of hyperedge orders.

    Args:
    hypergraph (dict): A dictionary representing a hypergraph, where keys are vertices
                       and values are sets of hyperedges.

    Returns:
    tuple: A tuple containing the max, average, minimum, and standard deviation of the
           set of hyperedge orders.
    """
    hyperedge_orders = []
    for edge_set in hypergraph.values():
        for edge in edge_set:
            hyperedge_orders.append(len(edge))

    max_order = max(hyperedge_orders)
    min_order = min(hyperedge_orders)
    avg_order = sum(hyperedge_orders) / len(hyperedge_orders)

    # Compute the standard deviation
    variance = sum((order - avg_order) ** 2 for order in hyperedge_orders) / len(hyperedge_orders)
    std_dev = np.sqrt(variance)

    return hyperedge_orders, max_order, min_order, avg_order, std_dev


def hyperedge_weight_stats(pstring_dict):
    """
    Compute statistics of distribution of edge weights in the Pauli hypergraph

    Args:
        Pauli string dictionary {"XXIX": 1.2, "XXZY": 0.4, ...}
    Returns:
        Statistics on Pauli hypergraph edge weights i.e., {c_i} for H = Σ c_i P_i
        int: max c_i
        int: min c_i
        float: mean c_i
        float: Std. dev. of distribution of c_i

    """
    weights = convert_pauli_hamiltonian(pstring_dict)[1][:]
    mean_weight = sum(abs(weight) for weight in weights) / len(weights)
    variance = sum((abs(weight) - mean_weight) ** 2 for weight in weights) / len(weights)
    std_dev = np.sqrt(variance)

    return weights, max(abs(weight) for weight in weights), min(abs(weight) for weight in weights), mean_weight, std_dev


def compute_graph_metrics(pstring_dict):
    """
        This function computes all graph metrics defined above
        and returns their values given a Pauli string
        stored in a dictionary format
    """
    pstrings = convert_pauli_hamiltonian(pstring_dict)[0][:]
    hypergraph = build_hypergraph(pstrings)

    edge_weight_stats = hyperedge_weight_stats(pstring_dict)
    edge_orders_stats = hyperedge_orders_stats(hypergraph)
    vertex_deg_stats = vertex_degree_stats(hypergraph)
    n_qubits = len(hypergraph)

    return n_qubits, vertex_deg_stats, edge_weight_stats, edge_orders_stats
