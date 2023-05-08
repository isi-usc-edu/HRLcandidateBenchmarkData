import glob
import json
from compute_metrics import compute_metrics, add_metrics
import pandas as pd
import matplotlib.pyplot as plt
from random_hamiltonians import generate_random_H
from FF_plus_impurity import generate_samples


def generate_dataframe():
    samples1 = glob.glob('free_fermion_stats/*_metrics.json')
    samples2 = glob.glob('random_H/*_metrics.json')
    samples3 = glob.glob('molecules/*/*_metrics.json')

    analysis1 = {}

    index = 0
    for sample in samples1:
        with open(sample) as user_file:
            f = user_file.read()
        metrics_dict = json.loads(f)
        analysis1[index] = metrics_dict
        index += 1
    keys = metrics_dict.keys()
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

    analysis3 = {}

    index = 0
    for sample in samples3:
        with open(sample) as user_file:
            f = user_file.read()
        metrics_dict = json.loads(f)
        analysis3[index] = metrics_dict
        index += 1

    df3 = pd.DataFrame.from_dict(analysis3).T

    return df1, df2, df3, keys


def plot_metrics(df, axes):
    ax = df.plot(kind="scatter", x=axes[0], y=axes[1], ec='r', fc='none', label='FF + Impurity')
    return ax


if __name__ == '__main__':
    print('Program started')

    ns = [i for i in range(10, 20, 1)]
    ms = [i for i in range(2, 5)]

    sample_id = 0
    for n in ns:
        for m in ms:
            if n > m:
                for i in range(20):
                    # Take 20 samples for each configuration

                    generate_samples(n, m, sample_id, num_gates=20)

                    # H_jw = generate_random_H(n,m)
                    # compute_metrics(H_jw, sample_id)
                    sample_id += 1

    df1, df2, df3, keys = generate_dataframe()

    axes = list(keys)

    for i in range(len(axes)):
        # s = pd.Series(['c', 'y'], index=[1, 0])
        if axes[i] != 'clustering_coefficient':
            ax = plot_metrics(df1, ["induced_norm", axes[i]])
            df2.plot(kind="scatter", x="induced_norm", y=axes[i], ec='g', fc='none', ax=ax, label="Random H")
            df3.plot(kind="scatter", x="induced_norm", y=axes[i], ec='b', fc='none', ax=ax,
                     label="Single Ref. Molecules")
            plt.legend()

            plt.savefig("induced_norm" + '_vs_' + axes[i] + '.png')
