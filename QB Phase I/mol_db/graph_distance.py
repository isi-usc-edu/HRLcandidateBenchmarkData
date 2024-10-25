import scipy.stats as stats
from compute_graph_metrics import *

def descriptive_statistics(data):
    """Calculate descriptive statistics for a given dataset."""
    descriptive_stats = {
        'mean': np.mean(data),
        'min': np.min(data),
        'max': np.max(data),
        'std_dev': np.std(data)
    }
    return descriptive_stats

def compare_statistics(stats1, stats2):
    """Prints comparative statistics for two datasets."""
    comparisons = {}
    for key in stats1:
        comparisons[key] = f"{key.capitalize()}: Graph 1 = {stats1[key]}, Graph 2 = {stats2[key]}"
    return comparisons

def perform_t_test(data1, data2):
    """Performs a T-test for two independent samples."""
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)  # Welch's t-test
    return t_stat, p_value

def test():
    # Example data for two graphs
    data_graph1 = np.random.normal(50, 10, 100)  # Synthetic data for graph 1
    eps = .2 * np.random.rand(100)

    # Perform statistical test
    t_stat, p_value = perform_t_test(data_graph1, data_graph1 + eps)

    return t_stat, p_value