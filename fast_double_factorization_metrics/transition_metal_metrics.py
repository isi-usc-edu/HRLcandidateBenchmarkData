from closedfermion.Transformations.fermionic_encodings import fermion_to_qubit_transformation
from closedfermion.Transformations.quartic_dirac_transforms import majorana_operator_from_quartic, double_factorization_from_quartic
from faux_ham import *
import time

if __name__ == "__main__":
    filenames = get_chk_filenames("chks/")


    chks=[]
    for chk_file in filenames:
        if '.chkfile' in chk_file:
            chks.append(chk_file)

    fname = chks[0]

    start_time = time.time()

    mol, _, hcore, norb, eri_4d = load_chk(fname)


    quartic_fermion = QuarticDirac(eri_4d, hcore, norb)
    majorana_op = majorana_operator_from_quartic(quartic_fermion)
    pauli_op = fermion_to_qubit_transformation(majorana_op, 'Jordan-Wigner')

    pauli_data = pauli_op.data
    vertex_degree_stats, weight_stats, edge_order_stats = compute_hypergraph_metrics(pauli_data)
    metrics_target = {}
    for key, val in list(vertex_degree_stats.items()) + list(weight_stats.items()) + list(
            edge_order_stats.items()):
        metrics_target[key] = val

    metrics_target['number_of_terms'] = len(pauli_data.keys())


    H_DF = double_factorization_from_quartic(quartic_fermion)
    eigs = np.array(truncate_df_eigenvalues(H_DF.eigs))
    print(len(eigs))

    end_time = time.time()

    print(f"Wall-clock time: {end_time-start_time}")