"""Helper functions for preprocessing."""
import numpy as np


def crop_hypno(hypno, crop=15):
    """Crop hypnogram."""
    # We keep up to 15 minutes before / after sleep
    start_to_firstsleep_min = np.nonzero(hypno)[0][0] / 2
    lastsleep = np.nonzero(hypno)[0][-1]
    lastsleep_to_end_min = (len(hypno) - lastsleep) / 2
    tmin, tmax = 0, None  # must be in seconds
    if start_to_firstsleep_min > crop:
        tmin = (start_to_firstsleep_min - crop) * 60
    if lastsleep_to_end_min > crop:
        tmax = lastsleep * 30 + crop * 60
    if tmax is None:
        hypno = hypno[int(tmin / 60 * 2):]
    else:
        hypno = hypno[int(tmin / 60 * 2):int(tmax / 60 * 2)]
    return hypno, tmin, tmax


def consensus_scores(hypnos, idx_best_hypno):
    """Create a consensus scores with majority voting.

    When ties are present, use the best scorer of the record.

    For more details, see Guillot et al 2020.
    """
    from scipy.stats import mode
    # Initialize output array
    hyp_cons = np.zeros_like(hypnos[0, :])

    for i, _ in enumerate(hyp_cons):
        mod, count = mode(hypnos[:, i])
        mod, count = mod[0], count[0]
        n_unique = len(np.unique(hypnos[:, i]))
        # Deal with ties
        if count == 2 and n_unique == 3:
            # [0, 0, 1, 1, 2] - n_unique = 3, tie, take best scorer
            # [0, 0, 1, 2, 3] - n_unique = 4, no tie
            mod = hypnos[idx_best_hypno, i]
        hyp_cons[i] = mod

    return hyp_cons
