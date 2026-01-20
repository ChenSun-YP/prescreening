import math
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
import glob
import pandas as pd


# check all intervals on base of the histogram is full
def check_correlation_filled_bins(
    pkl_path="data/analysis/selected_neurons_first_200s/crosscorrs_edge_mean_True_ultra-fine.pkl",  # find a way to automatically get this path
    out_dir="selected_neurons_first_200s",  # for debugging purposes
    harshness=0.10,  # at most 10% of bins can be empty
    power_threshold=-10,  # bins with value < e^-10 are considered empty
):
    """
    Saves (pre, post) pairs where at least `fraction_threshold` of corr bins
    have |corr| < e^(power_threshold).

    Parameters
    ----------
    pkl_path : str
        Path to crosscorrs pickle file
    fraction_threshold : float
        Fraction of bins required (e.g. 0.10 = 10%)
    power_threshold : float
        Power for exponential threshold (e.g. -10 → e^-10)
    """

    os.makedirs(out_dir, exist_ok=True)

    bad_pairs_path = os.path.join(out_dir, "bad_pairs.txt")
    good_pairs_path = os.path.join(out_dir, "good_pairs.txt")

    threshold = math.exp(power_threshold)

    with open(pkl_path, "rb") as f:
        crosscorrs = pickle.load(f)

    bad_pairs = []
    good_pairs = []

    for (pre, post), value in crosscorrs.items():
        # value is a tuple:
        # (lags, corr, mean, std, scores, total_score)

        # print(f"Filtering pair: {pre}, {post}")

        corr = value[1]

        if corr is None or len(corr) == 0:
            continue

        max_small = len(corr) * harshness
        # print(f"Max small bins allowed: {max_small}")

        num_small = np.sum(np.abs(corr) < threshold)
        # print(f"Number of small bins: {num_small}")

        if num_small >= max_small:
            # print(
            #     f"{pre}, {post} — " f"{num_small} of bins |corr| < e^{power_threshold}"
            # )
            bad_pairs.append((pre, post))
        else:
            good_pairs.append((pre, post))

        # Write bad pairs
    with open(bad_pairs_path, "w") as f:
        for pre, post in bad_pairs:
            f.write(f"{pre}\t{post}\n")

    # Write good pairs
    with open(good_pairs_path, "w") as f:
        for pre, post in good_pairs:
            f.write(f"{pre}\t{post}\n")


def filter_pairs_using_correlation_filled_bins(
    pairs,
    bad_pairs_path="selected_neurons_first_200s\\bad_pairs.txt",
):
    """
    Remove any pair that appears in bad_pairs.txt
    from the provided list of pairs.

    Parameters
    ----------
    pairs : list of tuple
        Original list of (pre, post) pairs
    good_pairs_path : str
    bad_pairs_path : str

    Returns
    -------
    filtered_pairs : list of tuple
    """

    def load_pairs(path):
        loaded = set()
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                pre, post = line.strip().split()
                loaded.add((pre, post))
        return loaded

    bad_pairs = load_pairs(bad_pairs_path)
    # print("bad_pairs", bad_pairs)

    filtered_pairs = [pair for pair in pairs if pair not in bad_pairs]

    return filtered_pairs


def check_unimodality(crosscorr, tol=1e-5):
    """
    Check if the cross-correlogram is unimodal.

    Parameters
    ----------
    crosscorr : np.ndarray
        The cross-correlogram to check.
    tol : float
        Tolerance for determining peaks.

    Returns
    -------
    bool
        True if unimodal, False otherwise.
    """
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(crosscorr, height=tol)
    return len(peaks) <= 1
