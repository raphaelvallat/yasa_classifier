"""Helper functions."""
import os
import numpy as np
import pandas as pd
from scipy.stats import iqr, mode
from sklearn.metrics import accuracy_score

NUM2STR = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
STR2NUM = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}


def perc_transition(x):
    """Percentage of transitions in sleep staging."""
    return 100 * (x != x.shift(1)).sum() / x.shape[0]


def mean_std(x):
    """Return the mean and standard deviation."""
    return f"{x.mean():.2f} ± {x.std():.2f}"


def median_iqr(x):
    """Return the median and IQR."""
    return f"{x.median():.2f} ± {iqr(x):.2f}"


def consensus_score(data):
    """Create an unbiased consensus score on N-1 scorers.

    When ties are present (e.g. [0, 0, 1, 1]), use the scoring of the
    most reliable scorer of the record, i.e. the one with the overall strongest
    agreement with all the other ones.

    Currently only supports up to 5 scorers (ties can occur with 4 or 5 scorers).

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe where each column is a scorer.
    """
    # Reset index so that .loc = .iloc
    data = data.reset_index(drop=True)  # Also copy the data
    n_scorers = data.shape[1]
    # Convert to INT if scores are in STR
    if (data.dtypes == "object").all():
        orig_dtype = "object"
        data = data.replace(STR2NUM).astype(int)
    elif (data.dtypes == "int").all():
        orig_dtype = "int"
    else:
        raise ValueError("Dtype not recognized. Must be object or INT.")
    # Calculate pairwise agreement between scorer
    corr_acc = data.corr(accuracy_score).mean()
    # Find index of best scorer
    idx_best_scorer = corr_acc.sort_values(ascending=False).index[0]
    # Calculate consensus stage for each epoch
    mod, counts = mode(data, axis=1)
    mod = np.squeeze(mod)
    counts = np.squeeze(counts)
    n_unique = data.nunique(1).to_numpy()
    # Find indices of ties and replace values by most reliable scorer
    if n_scorers in [2, 3]:
        pass
    elif n_scorers == 4:
        # [0, 0, 1, 1]
        ties = np.where(counts == 2)[0]
        mod[ties] = data.loc[ties, idx_best_scorer].to_numpy()
    elif n_scorers == 5:
        # [0, 0, 1, 1, 2] - n_unique = 3, tie, take best scorer
        # [0, 0, 1, 2, 3] - n_unique = 4, no tie
        ties = np.where((counts == 2) & (n_unique == 3))[0]
        mod[ties] = data.loc[ties, idx_best_scorer].to_numpy()
    else:
        raise ValueError("%i scorers not supported." % n_scorers)

    # Convert back to original dtype
    if orig_dtype == "object":
        mod = pd.Series(mod).replace(NUM2STR).to_numpy()
    return mod

