import numpy as np

def time_rescaling_ks(Pb, y):
    """
    Applies time rescaling transformation and performs the KS test.

    Args:
        Pb: (T,) Estimated probability per bin from the fitted GLM.
        y:  (T,) Binary spike train (0 or 1).

    Returns:
        KS_score: Kolmogorov-Smirnov statistic.
        Zs: Sorted transformed values expected to be uniform.
        b: Theoretical uniform CDF values.
        b95: 95% confidence interval bounds for uniform CDF.
        Tau: Rescaled interspike intervals.
    """
    # Find spike times
    T_time = np.where(y > 0)[0]
    if len(T_time) < 2:
        raise ValueError("Not enough spikes to perform time rescaling.")

    # Compute rescaled intervals
    q = -np.log(1 - Pb)
    Tau = np.zeros(len(T_time) - 1)
    
    for i in range(1, len(T_time)):
        r = np.random.uniform()
        Tau[i - 1] = np.sum(q[(T_time[i-1] + 1):(T_time[i])]) - np.log(1 - r * (1 - np.exp(-q[T_time[i]])))

    # Transform to uniform distribution
    Z = 1 - np.exp(-Tau)
    Zs = np.sort(Z)

    # Compute empirical CDF
    n = len(Zs)
    b = np.linspace(1/n, 1, n)  # Uniform CDF
    b95 = np.column_stack([b - 1.36 / np.sqrt(n), b + 1.36 / np.sqrt(n)])

    # KS score
    KS_score = np.max(np.abs(Zs - b)) / np.max(np.abs(b95[:, 0] - b))

    return KS_score, Zs, b, b95, Tau
