import os
import glob
from pyscf import gto, scf, lib
from openfermion import MolecularData, get_fermion_operator
from openfermionpyscf import generate_molecular_hamiltonian
from openfermion.transforms import jordan_wigner
from main import parse_line, input

basis_sets = {'vdz': 'cc-pVDZ', 'vtz': 'cc-pVTZ', 'vqz': 'cc-pVQZ', 'v5z': 'cc-pV5Z'}



def load_chk_files(directory):
    chk_files = glob.glob(os.path.join(directory, '*.chk')) + glob.glob(os.path.join(directory, '*.chkfile'))
    return chk_files


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


def main(directory, geos):
    chk_files = load_chk_files(directory)


    for chk_file in chk_files:
        mf, mol = load_pyscf_molecule(chk_file)
        idx=0
        basis_set = None
        if 'cc-pV' in chk_file:
            idx = chk_file.index('cc-pV')
            sys = chk_file[2:idx - 1]
            basis_set = chk_file[idx: idx + 7]


        elif 'v' in chk_file:
            idx = chk_file.index('v')
            sys = chk_file[2:idx - 1]
            basis_set_a = None
            for key in basis_sets.keys():
                if key in chk_file:
                    basis_set_a = basis_sets[key]
            basis_set = basis_set_a

        if sys == 'c' and basis_set == 'cc-pVDZ':
            geo = geos[sys]['atoms']

            h_core = mf.get_hcore()  # One-electron integrals
            eri = mf.mol.intor('int2e')

            multiplicity = mf.mol.spin + 1

            # Generate and populate instance of MolecularData.
            molecule = MolecularData(geo, basis_set, multiplicity, description="1.45")

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
            print(Hjw)
            print(mf.e_tot)
            break


if __name__ == "__main__":
    directory = './'  # Change this to your directory containing .chk or .chkfile files
    chk_files = load_chk_files(directory)


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

    main(directory, geometries)
