from collections import defaultdict
import re
import hypernetx as hnx
import numpy as np
import itertools
import networkx as nx

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
        and indices k_j are those such that σ_ik_j != 1

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
    num_hyperedges = [len(hyperedges) for hyperedges in hypergraph.values()]
    max_hyperedges = max(num_hyperedges)
    avg_hyperedges = sum(num_hyperedges) / len(num_hyperedges)

    return max_hyperedges, avg_hyperedges


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
    hyperedge_orders = [len(hyperedge) for hyperedges in hypergraph.values() for hyperedge in hyperedges]
    max_order = max(hyperedge_orders)
    min_order = min(hyperedge_orders)
    avg_order = sum(hyperedge_orders) / len(hyperedge_orders)

    # Compute the standard deviation
    variance = sum((order - avg_order) ** 2 for order in hyperedge_orders) / len(hyperedge_orders)
    std_dev = np.sqrt(variance)

    return max_order, avg_order, min_order, std_dev


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

    return max(abs(weight) for weight in weights), min(abs(weight) for weight in weights), mean_weight, std_dev


def build_weighted_hypernetx_graph(pauli_strings, edge_weights, vertex_offset=1000):
    edge_dict = {}

    for idx, pauli_string in enumerate(pauli_strings):
        hyperedge = pauli_string_to_hyperedge(pauli_string)
        if hyperedge:
            edge_dict[idx] = {vertex + vertex_offset for vertex in hyperedge}  # Offset vertex indices

    try:
        H = hnx.Hypergraph(edge_dict)
    except Exception as e:
        print("Error occurred while creating the HyperNetX graph:", e)
        return None

    for idx, weight in enumerate(edge_weights):
        if idx in H.edges:
            H.edges[idx].weight = weight

    return H


def node_strengths(H):
    """
    Computes the strength of each node in a weighted HyperNetX hypergraph.

    Args:
        H (hnx.Hypergraph): A weighted HyperNetX hypergraph object.

    Returns:
        dict: A dictionary with nodes as keys and their strengths as values.
    """
    strengths = {node: 0 for node in H.nodes}

    for edge in H.edges:
        weight = H.edges[edge].weight
        for node in H.edge_to_nodes(edge):
            strengths[node] += weight

    return strengths


def weighted_dual_hypergraph(H):
    """
    Computes the weighted dual hypergraph of a given weighted HyperNetX hypergraph.

    Args:
        H (hnx.Hypergraph): A weighted HyperNetX hypergraph object.

    Returns:
        hnx.Hypergraph: The weighted dual hypergraph.
    """
    edge_dict = {edge: H.edge_to_nodes(edge) for edge in H.edges}
    dual_edge_dict = {node: set() for node in H.nodes}

    for edge in H.edges:
        weight = H.edges[edge].weight
        for node in H.edge_to_nodes(edge):
            dual_edge_dict[node].add(edge)

    dual_H = hnx.Hypergraph(dual_edge_dict)

    for edge in dual_H.edges:
        dual_H.edges[edge].weight = H.nodes[edge].weight

    return dual_H


def weighted_hypergraph_to_clique_expansion(H):
    G = nx.Graph()

    for edge in H.edges:
        weight = H.edges[edge]['weight'] if H.edges[edge]['weight'] is not None else 1  # Ensure weight is never None
        vertices = list(H.edges[edge].elements)  # Access the vertices of the edge
        for v1, v2 in itertools.combinations(vertices, 2):
            if G.has_edge(v1, v2):
                G[v1][v2]['weight'] += weight
            else:
                G.add_edge(v1, v2, weight=weight)

    return G


def weighted_hypergraph_clustering_coefficient(H):
    """
    Computes the average weighted clustering coefficient for a weighted HyperNetX hypergraph.

    Args:
        H (hnx.Hypergraph): A weighted HyperNetX hypergraph object.

    Returns:
        float: The average weighted clustering coefficient of the hypergraph.
    """
    G = weighted_hypergraph_to_clique_expansion(H)
    clustering_coeffs = nx.clustering(G, weight="weight").values()
    avg_clustering_coeff = sum(clustering_coeffs) / len(clustering_coeffs)

    return avg_clustering_coeff


def weighted_hypergraph_degrees(H):
    """
    Computes the weighted degree of each node in a weighted HyperNetX hypergraph.

    Args:
        H (hnx.Hypergraph): A weighted HyperNetX hypergraph object.

    Returns:
        dict: A dictionary with nodes as keys and their weighted degrees as values.
    """
    G = weighted_hypergraph_to_clique_expansion(H)
    degrees = dict(G.degree(weight="weight"))

    return degrees


def centrality_graph_metrics(H):
    G = weighted_hypergraph_to_clique_expansion(H)
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    return degree_centrality, closeness_centrality, betweenness_centrality


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

    #edge_weights = convert_pauli_hamiltonian(pstring_dict)[1][:]

    # H_weighted = build_weighted_hypernetx_graph(pstrings, edge_weights)

    # Compute hypergraph metrics with HyperNetX
    # c_degree, c_closeness, c_betweenness = centrality_graph_metrics(H_weighted)

    # clustering_coeff = weighted_hypergraph_clustering_coefficient(H_weighted)

    return n_qubits, vertex_deg_stats, edge_weight_stats, edge_orders_stats
