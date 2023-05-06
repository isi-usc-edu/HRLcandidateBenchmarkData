from collections import defaultdict
import json
import re
import hypernetx as hnx
import numpy as np
import itertools
import networkx as nx


def compute_metrics(H_jw, sample_id):
    pstring_dict = {}
    for ops in H_jw:
        pstring = str(ops)
        index = pstring.index("[")
        p_key = pstring[index:]
        # p_value = complex(pstring[1:index-2])
        p_value = complex(pstring[:index])
        pstring_dict[p_key] = p_value

    metrics = {}
    metrics["induced_norm"] = H_jw.induced_norm(order=1)

    # Compute graph based metrics
    total_pstrings, n_qubits, vertex_deg_stats, edge_weight_stats, edge_order_stats, \
        c_degree, c_closeness, c_betweenness, clustering_coeff = compute_graph_metrics(pstring_dict)

    metrics['total_pstrings'] = total_pstrings

    metrics['max_vertex_degree'] = vertex_deg_stats[0]
    metrics['avg_vertex_degree'] = vertex_deg_stats[1]

    metrics['max_edge_weight'] = edge_weight_stats[0]
    metrics['min_edge_weight'] = edge_weight_stats[1]
    metrics['avg_edge_weight'] = edge_weight_stats[2]
    metrics['edge_weight_std_dev'] = edge_weight_stats[3]

    metrics['max_edge_order'] = edge_order_stats[0]
    metrics['min_edge_order'] = edge_order_stats[1]
    metrics['avg_edge_order'] = edge_order_stats[2]
    metrics['edge_order_std_dev'] = edge_order_stats[3]

    metrics['n_qubits'] = n_qubits

    metrics["centrality_degree"] = c_degree
    metrics["centrality_closeness"] = c_closeness
    metrics["centrality_betweenness"] = c_betweenness

    metrics["clustering_coefficient"] = clustering_coeff

    path = "random_H/"  # "free_fermion_stats/"
    with open(path + str(sample_id) + "_metrics.json", "w") as write_file:
        json.dump(metrics, write_file, indent=4, sort_keys=True)

    return metrics


def convert_pauli_hamiltonian(hamiltonian):
    """
        A function for converting a Pauli string Hamiltonian stored in a dictionary format
        to a nested list format used by the functions for computing Trotter spectral error bounds
        and the graph metrics
    """
    pauli_strings = []
    coefficients = []

    max_index_global = 0

    for key, value in hamiltonian.items():
        if key == "[]":
            continue

        terms = key.split()
        term_data = [(term[0], int(re.search(r'\d+', term).group())) for term in terms if re.search(r'\d+', term)]
        max_index = max([index for _, index in term_data])
        max_index_global = max(max_index_global, max_index)

    for key, value in hamiltonian.items():
        if key == "[]":
            continue

        terms = key.split()
        term_data = [(term[0], int(re.search(r'\d+', term).group())) for term in terms if re.search(r'\d+', term)]

        formatted_key = ""
        for i in range(max_index_global + 1):
            current_term = [term for term, index in term_data if index == i]

            if len(current_term) > 0:
                formatted_key += current_term[0][0]
            else:
                formatted_key += "I"

        pauli_strings.append(formatted_key[1:])
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


def build_hypernetx_graph(pauli_strings):
    """
    Creates a HyperNetX graph from the given Pauli string hypergraph.

    Args:
        pauli_strings (list): A list of Pauli strings.

    Returns:
        hnx.Hypergraph: A HyperNetX hypergraph object.
    """
    # Use the previously defined pauli_string_to_hyperedge function to extract hyperedges
    hyperedges = [pauli_string_to_hyperedge(pauli_string) for pauli_string in pauli_strings]

    # Create a dictionary where keys are edge IDs and values are sets of vertices
    edge_dict = {i: set(hyperedge) for i, hyperedge in enumerate(hyperedges)}

    # Create a HyperNetX graph from the edge dictionary
    H = hnx.Hypergraph(edge_dict)

    return H


def build_weighted_hypernetx_graph(pauli_strings, edge_weights):
    """
    Creates a HyperNetX graph from the given Pauli string hypergraph with the given edge weights.

    Args:
        pauli_strings (list): A list of Pauli strings.
        edge_weights (list): A list of edge weights corresponding to each Pauli string.

    Returns:
        hnx.Hypergraph: A HyperNetX hypergraph object.
    """
    if len(pauli_strings) != len(edge_weights):
        raise ValueError("The lengths of pauli_strings and edge_weights must be equal.")

    # Use the previously defined pauli_string_to_hyperedge function to extract hyperedges
    hyperedges = [pauli_string_to_hyperedge(pauli_string) for pauli_string in pauli_strings]

    # Create a dictionary where keys are edge IDs and values are sets of vertices
    edge_dict = {i: set(hyperedge) for i, hyperedge in enumerate(hyperedges)}

    # Create a HyperNetX graph from the edge dictionary
    H = hnx.Hypergraph(edge_dict)

    # Assign edge weights to the hyperedges
    for i, weight in enumerate(edge_weights):
        H.edges[i].weight = weight

    return H


def weighted_hypergraph_to_clique_expansion(H):
    """
    Transforms a weighted HyperNetX hypergraph into a weighted NetworkX graph using the clique expansion method.

    Args:
        H (hnx.Hypergraph): A weighted HyperNetX hypergraph object.

    Returns:
        nx.Graph: A weighted NetworkX graph object.
    """
    G = nx.Graph()

    for edge in H.edges:
        vertices = list(H.edge_to_nodes(edge))
        weight = H.edges[edge].weight
        for pair in itertools.combinations(vertices, 2):
            if G.has_edge(*pair):
                G[pair[0]][pair[1]]["weight"] += weight
            else:
                G.add_edge(pair[0], pair[1], weight=weight)

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

    total_strings = len(pstrings)
    edge_weight_stats = hyperedge_weight_stats(pstring_dict)
    edge_orders_stats = hyperedge_orders_stats(hypergraph)
    vertex_deg_stats = vertex_degree_stats(hypergraph)
    n_qubits = len(hypergraph)

    edge_weights = convert_pauli_hamiltonian(pstring_dict)[1][:]
    H_weighted = build_weighted_hypernetx_graph(pstrings, edge_weights)

    # Compute hypergraph metrics with HyperNetX
    c_degree, c_closeness, c_betweenness = centrality_graph_metrics(H_weighted)

    clustering_coeff = weighted_hypergraph_clustering_coefficient(H_weighted)

    return total_strings, n_qubits, vertex_deg_stats, edge_weight_stats, edge_orders_stats,\
            c_degree, c_closeness, c_betweenness, clustering_coeff
