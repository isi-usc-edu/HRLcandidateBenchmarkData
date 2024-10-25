import glob
import json
from pyscf import gto, scf, lib
from openfermion import MolecularData, get_fermion_operator
from openfermion.transforms import jordan_wigner
from main import parse_line, input
from compute_graph_metrics import compute_graph_metrics
import numpy as np

basis_sets = {'vdz': 'cc-pVDZ', 'vtz': 'cc-pVTZ', 'vqz': 'cc-pVQZ', 'v5z': 'cc-pV5Z'}


def load_pyscf_molecule(chk_file):
    mol = lib.chkfile.load_mol(chk_file)
    mf = scf.RHF(mol)
    mf.__dict__.update(lib.chkfile.load(chk_file, 'scf'))
    return mf, mol


def save_pauli_dict(pauli_hamiltonian, output_file):
    pauli_dict = {}
    for term in pauli_hamiltonian.terms:
        pauli_str = tuple(term)
        coefficient = pauli_hamiltonian.terms[term]
        pauli_dict[pauli_str] = coefficient.real

    with open(output_file, 'w') as f:
        for key, value in pauli_dict.items():
            f.write(f"{key} : {value}\n")


def parse_filenames(filenames):
    systems = []
    for filename in filenames:
        # Remove the file extension first
        clean_filename = filename.replace('_chkfile.chk', '')
        clean_filename = "chks/" + clean_filename

        # Split by last underscore to separate the basis set
        parts = clean_filename.rsplit('_', 1)
        if len(parts) < 2:
            continue  # skip if no basis set is found

        system_name, basis_set = parts[0], parts[1]

        # Create a dictionary for the system
        system_info = {
            'System Name': system_name[5:],
            'Basis Set': basis_set,
            'Filename': filename
        }

        systems.append(system_info)

    return systems


def main(systems, geos, targ_sys, targ_basis="cc-pVDZ"):
    for system in systems:
        sys, basis_set, filename = system['System Name'], system['Basis Set'], system['Filename']

        mf, mol = load_pyscf_molecule("chks/" + filename)

        if sys == targ_sys and basis_set == targ_basis:
            geo = geos[sys]['atoms']

            geo2 = []
            for i in range(len(geo)):
                geo2.append((geo[i][0].capitalize(), geo[i][1], geo[i][2], geo[i][3]))

            h_core = mf.get_hcore()  # One-electron integrals
            eri = mf.mol.intor('int2e')

            multiplicity = mf.mol.spin + 1

            # Generate and populate instance of MolecularData.
            molecule = MolecularData(geo2, basis_set, multiplicity)

            # Set the molecular data
            molecule.hf_energy = mf.e_tot
            molecule.nuclear_repulsion = mol.energy_nuc()
            molecule.one_body_integrals = h_core
            molecule.two_body_integrals = eri
            molecule.orbital_energies = mf.mo_energy
            molecule.canonical_orbitals = mf.mo_coeff
            molecule.overlap_integrals = mf.get_ovlp()

            # Get the Hamiltonian in an active space.
            molecular_hamiltonian = molecule.get_molecular_hamiltonian()
            Hjw = jordan_wigner(get_fermion_operator(molecular_hamiltonian))
            return Hjw, sys, basis_set


def get_chks_and_geos():
    directory = './'  # Change this to your directory containing .chk or .chkfile files
    glob_files = glob.glob('chks/*.chk') + glob.glob('chks/*.chkfile')

    # Removing 'chks/' prefix and storing results in a new array
    chk_files = [filename.split('chks/')[1] for filename in glob_files]

    # Splitting the input string by lines
    lines = input().strip().split('\n')

    # Initialize an empty dictionary to store the geometries
    geometries = {}

    # Process each line and store the data in the dictionary
    current_system = None
    for line in lines:
        parsed_data = parse_line(line)
        if parsed_data:
            current_system = parsed_data[0]
            geometries[current_system] = {
                'spin': parsed_data[1],
                'atoms': parsed_data[2]
            }
        else:
            if current_system:
                # Append additional atoms to the current system
                coordinates = line.split()
                atoms = [
                    (
                        coordinates[i], float(coordinates[i + 1]), float(coordinates[i + 2]), float(coordinates[i + 3]))
                    for i in range(0, len(coordinates), 4)]
                geometries[current_system]['atoms'].extend(atoms)
    return chk_files, geometries


def get_metrics(Hjw, sys, save=False):
    pstring_dict = {}

    for ops in Hjw:
        pstring = str(ops)
        index1 = pstring.index("[")
        index2 = pstring.index("]")
        p_key = pstring[index1:index2 + 1]
        p_value = complex(pstring[:index1])

        pstring_dict[p_key] = p_value

    metrics = {}
    metrics["induced_norm"] = Hjw.induced_norm(order=1)  # higher order norms?

    induced_norm = 0.
    for c in pstring_dict.values():
        induced_norm += np.abs(c)
    metrics["induced_norm"] = induced_norm

    # Compute graph based metrics
    n_qubits, vertex_deg_stats, edge_weight_stats, edge_order_stats = compute_graph_metrics(pstring_dict)

    metrics['total_pstrings'] = len(list(pstring_dict.keys()))

    """
    deg_data = vertex_deg_stats[0]
    weight_data = edge_weight_stats[0]
    order_data = edge_order_stats[0]
    """

    ## TO-DO add vertex degree std dev.
    metrics['max_vertex_degree'] = vertex_deg_stats[1]
    metrics['min_vertex_degree'] = vertex_deg_stats[2]
    metrics['avg_vertex_degree'] = vertex_deg_stats[3]
    metrics['vertex_degree_std_dev'] = vertex_deg_stats[4]

    # Full statistics
    metrics['max_edge_weight'] = edge_weight_stats[1]
    metrics['min_edge_weight'] = edge_weight_stats[2]
    metrics['avg_edge_weight'] = edge_weight_stats[3]
    metrics['edge_weight_std_dev'] = edge_weight_stats[4]

    metrics['max_edge_order'] = edge_order_stats[1]
    metrics['min_edge_order'] = edge_order_stats[2]
    metrics['avg_edge_order'] = edge_order_stats[3]
    metrics['edge_order_std_dev'] = edge_order_stats[4]

    metrics['n_qubits'] = n_qubits

    metrics['name'] = sys
    if save:
        with open(metrics['name'] + "_metrics.json", "w") as write_file:
            json.dump(metrics, write_file, indent=4, sort_keys=True)
    return metrics


def compute_pstring_stats(targ_sys):
    chk_files, geometries = get_chks_and_geos()
    systems = parse_filenames(chk_files)

    # Compute JW Hamiltonian and compute interaction hypergraph (stats) metrics
    Hjw, sys, basis = main(systems, geometries, targ_sys=targ_sys)

    metrics = get_metrics(Hjw, sys + "_" + basis, save=False)
    return metrics


if __name__ == "__main__":
    # Get chkfiles
    targ_sys = "p"
    metrics = compute_pstring_stats(targ_sys)

    for key, val in metrics.items():
        print(key, val)
        print(" ")
