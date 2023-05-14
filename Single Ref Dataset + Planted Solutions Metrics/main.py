import glob
import json
from compute_metrics import compute_metrics, add_metrics, compute_cas_metrics, add_metrics_CAS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from random_hamiltonians import generate_random_H
from FF_plus_impurity import generate_random_unitary_circuit
import cirq
from FF_plus_impurity import generate_samples
from compute_graph_metrics import *

def generate_fci_dataframe():
    fci_path = 'benchmarking_results/fci_energies.json'
    ccsd_path = 'benchmarking_results/ccsd_energies.json'

    with open(ccsd_path) as user_file:
        f = user_file.read()
    ccsd = json.loads(f)

    with open(fci_path) as user_file:
        g = user_file.read()
    fci = json.loads(g)

    fci_chem = list(fci.keys())
    ccsd_chem = list(ccsd.keys())

    samples = glob.glob('molecules/*/*_metrics.json')

    analysis = {}

    index = 0
    for sample in samples:

        with open(sample) as user_file:
            h = user_file.read()
        metrics_dict = json.loads(h)

        path = sample[sample.index('/')+1:]
        j = path.index('/')
        name = path[:j]

        metrics_dict["fci"] = False
        metrics_dict["ccsd"] = False

        if name in fci_chem:
            metrics_dict["fci"] = True
            if name in ccsd_chem:
                metrics_dict["ccsd"] = True

        metrics_dict["name"] = name

        analysis[index] = metrics_dict
        index += 1

    df = pd.DataFrame.from_dict(analysis).T

    return df


def generate_dataframe():
    samples1 = glob.glob('free_fermion_stats/*_metrics.json')
    samples2 = glob.glob('random_H/*_metrics.json')
    samples3 = glob.glob('cas_molecules/*/*_CAS_metrics.json')
    samples4 = glob.glob('molecules/*/*_metrics.json')
    samples5 = ['1water_CAS_metrics.json', '2water_CAS_metrics.json']

    analysis1 = {}

    index = 0
    for sample in samples1:
        with open(sample) as user_file:
            f = user_file.read()
        metrics_dict = json.loads(f)
        analysis1[index] = metrics_dict
        index += 1
    df1 = pd.DataFrame.from_dict(analysis1).T


    analysis2 = {}

    index = 0
    for sample in samples2:
        with open(sample) as user_file:
            f = user_file.read()
        metrics_dict = json.loads(f)
        analysis2[index] = metrics_dict
        index += 1

    df2 = pd.DataFrame.from_dict(analysis2).T

    analysis4 = {}

    index = 0
    for sample in samples4:
        with open(sample) as user_file:
            f = user_file.read()
        metrics_dict = json.loads(f)
        analysis4[index] = metrics_dict
        index += 1
    df4 = pd.DataFrame.from_dict(analysis4).T

    analysis3 = {}

    index = 0
    for sample in samples3:
        with open(sample) as user_file:
            f = user_file.read()
        metrics_dict = json.loads(f)
        analysis3[index] = metrics_dict
        index += 1

    df3 = pd.DataFrame.from_dict(analysis3).T

    analysis5 = {}

    index = 0
    for sample in samples5:
        with open(sample) as user_file:
            f = user_file.read()
        metrics_dict = json.loads(f)
        analysis5[index] = metrics_dict
        index += 1
    df5 = pd.DataFrame.from_dict(analysis5).T


    return df1, df2, df3, df4, df5


def plot_metrics(df, axes):
    ax = df.plot(kind="scatter", x=axes[0], y=axes[1], ec='r', fc='none', label="Single Ref. Molecules")
    return ax


if __name__ == '__main__':
    print('Program started')

    ns = [i for i in range(4, 50, 2)]
    ms = [i for i in range(2, 20)]


    a = 100
    b = 100

    sample_id = 0
    for n in ns:
        for m in ms:
            if n > m:
                H_jw = generate_random_H(n,m)
                compute_metrics(H_jw, sample_id, 0)

                sample_id += 1
    """
    
    sample_id = 0
    for n in ns:
        for m in ms:
            if n > m:
                H_jw = generate_samples(n, m, a, b)

                compute_metrics(H_jw, sample_id, 1)

                sample_id += 1

    """
    df1, df2, df3, df4, df5 = generate_dataframe()


    axes = ['total_pstrings', 'max_vertex_degree', 'avg_vertex_degree',
            'max_edge_weight', 'min_edge_weight', 'avg_edge_weight',
            'edge_weight_std_dev', 'max_edge_order', 'min_edge_order', 'avg_edge_order',
            'edge_order_std_dev', 'n_qubits', 'induced_norm']

    axes_dict = {0 : 'Total Pauli Strings',1: 'Max Vertex Degree',2: 'Average Vertex Degree',
            3: 'Max '+ r'$|w_i|$', 4:'Min ' + r'$|w_i|$', 5:'Average ' + r'$|w_i|$',
            6: r'$|w_i|$' + r' $\sigma$', 7: 'Max Hyperedge Order',8: 'Min Hyperedge Order', 9: 'Average Hyperedge Order',
            10: 'Hyperedge Order' r' $\sigma$', 11: 'Number of Qubits', 12: r'$\sum_i |w_i|$'}
    # Get the full list of styles
    #sns.axes_style()

    custom = {"axes.edgecolor": "red", "grid.linestyle": "dashed", "grid.color": "black"}
    sns.set()
    sns.set_context("talk", rc={"lines.linewidth": 2.5})
    for i in range(len(axes)):
        if not axes[i] == 'n_qubits':
            ms = 50
            ax = df1.plot(kind="scatter", x="n_qubits", y=axes[i], color="purple", marker="D", label="FF + Imp. H", s=ms, figsize=(10, 10))

            df2.plot(kind="scatter", x="n_qubits", y=axes[i], color="green", marker='s', ax=ax, label="Random Pauli H",s=ms)

            df3.plot(kind="scatter", x="n_qubits", y=axes[i], color="orange", marker='o', ax=ax, label="CAS Pauli H",s=ms)

            #df4.plot(kind="scatter", x="n_qubits", y=axes[i], ec='purple', fc='none', ax=ax, label="Full Pauli H", s=ms)

            df5.plot(kind="scatter", x="n_qubits", y=axes[i],color="red", marker='^', ax=ax, label="Mg-rich Instance", s=ms)
            plt.legend(prop={"weight": "bold"})
            plt.xlabel('Number of Qubits', weight='bold')
            plt.ylabel(axes_dict[i], weight='bold')
            plt.yscale('log')


            plt.title(axes_dict[i]  + " vs. Number of Qubits", weight='bold')


            plt.savefig("n_qubits" + '_vs_' + axes[i] + '.png')




    """
    for i in range(len(axes)):

        ax = df1.plot(kind="scatter", x="n_qubits", y=axes[i], ec='r', fc='none', label="FF + Imp.")
        df2.plot(kind="scatter", x="n_qubits", y=axes[i], ec='b', fc='none', ax=ax, label="Random H")

        df3.plot(kind="scatter", x="n_qubits", y=axes[i], ec='g', fc='none', ax=ax, label="CAS Pauli H")
        plt.legend()
        plt.savefig("n_qubits" + '_vs_' + axes[i] + '.png')
    """


    #df = generate_fci_dataframe()
    #df["color"] = np.where(df["ccsd"] == True, "red", "blue")

    #df.plot(x='log_size', y='induced_norm', kind="scatter", color=df["color"])

    #df3.to_csv('cas_dataset.zip', index=False)

    #add_metrics_CAS()
    #print(df3)

    #plt.show()