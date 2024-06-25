import json
import glob
from scipy.special import comb as choose
from pyscf import gto
import numpy as np
from compute_graph_metrics import compute_graph_metrics


def compute_metrics(H_jw, sample_id, binary):
    pstring_dict = {}


    for ops in H_jw:
        pstring = str(ops)
        index1 = pstring.index("[")
        index2 = pstring.index("]")
        p_key = pstring[index1:index2 + 1]
        #index3 = pstring.index("(")
        #index4 = pstring.index(")")
        p_value = complex(pstring[:index1])

        pstring_dict[p_key] = p_value



    metrics = {}
    metrics["induced_norm"] = H_jw.induced_norm(order=1)

    # Compute graph based metrics
    n_qubits, vertex_deg_stats, edge_weight_stats, edge_order_stats = compute_graph_metrics(pstring_dict)

    metrics['total_pstrings'] = len(list(pstring_dict.keys()))

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

    if binary == 0:
        path = "random_H/"
    else:
        path = "free_fermion_stats/"

    with open(path + str(sample_id) + "_metrics.json", "w") as write_file:
        json.dump(metrics, write_file, indent=4, sort_keys=True)

    return metrics

def compute_cas_metrics(pstring_dict, path, name):


    metrics = {}


    induced_norm = 0.
    for c in pstring_dict.values():
        induced_norm += np.abs(c)
    metrics["induced_norm"] = induced_norm

    # Compute graph based metrics
    n_qubits, vertex_deg_stats, edge_weight_stats, edge_order_stats = compute_graph_metrics(pstring_dict)

    metrics['total_pstrings'] = len(list(pstring_dict.keys()))

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

    metrics['name'] = name

    with open(path+"_CAS_metrics.json", "w") as write_file:
        json.dump(metrics, write_file, indent=4, sort_keys=True)

    return metrics

def add_metrics_CAS():
    molecules = glob.glob('cas_molecules/*/')
    for molecule in molecules:
        metrics = glob.glob(molecule + "*_CAS_metrics.json")
        if metrics:
            index1 = molecule.index("/")
            index2 = molecule[index1 + 1:].index("/")
            name = molecule[index1 + 1:index1 + index2 + 1]
            f = open("CAS.json")
            cas = json.load(f)[name]


            ne, no = cas[0], cas[1]


            log_size = np.log(choose(no, ne)**2)

            f2 = open(metrics[0])
            metrics_new = json.load(f2)

            metrics_new['log_size'] = log_size

            with open(metrics[0], "w") as write_file:
                json.dump(metrics_new, write_file, indent=4, sort_keys=True)

def add_metrics():
    # open cas and CAS_metrics.json


    molecules = glob.glob('molecules/*/')
    for molecule in molecules:
        metrics = glob.glob(molecule + "*metrics.json")
        if metrics:
            index1 = molecule.index("/")
            index2 = molecule[index1 + 1:].index("/")
            name = molecule[index1 + 1:index1 + index2 + 1]
            molecule_data = glob.glob(molecule + name + '.json')[0]
            f = open(molecule_data)
            data = json.load(f)

            mol = gto.Mole()
            mol.atom = data['geometry']
            mol.basis = data['bases'][0]
            mol.charge = data['charge']
            mol.spin = data['multiplicity'] - 1
            mol.symmetry = False
            mol.build()

            ne, no = mol.nelec, mol.nao

            log_size = np.log(choose(no, ne[0]) * choose(no, ne[1]))

            f2 = open(metrics[0])
            metrics_new = json.load(f2)


            metrics_new['log_size'] = log_size
            

            with open(metrics[0], "w") as write_file:
                json.dump(metrics_new, write_file, indent=4, sort_keys=True)
