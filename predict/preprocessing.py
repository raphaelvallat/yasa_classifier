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
    
    Currently only 4, 5 scorers (DOD) or 6 scorers (IS-RC) are supported.
    The hypnograms should have the following stages: [0, 1, 2, 3, 4]
    
    Parameters
    ----------
    hypnos : np.array 
        The hypnograms, shape = (n_scorers, n_epochs)
    idx_best_hypno : int
        The index number (iloc) of the most reliable scorer
    """
    from scipy.stats import mode
    n_scorers = hypnos.shape[0]
    hyp_cons = np.zeros_like(hypnos[0, :])

    for i, _ in enumerate(hyp_cons):
        mod, count = mode(hypnos[:, i])
        mod, count = mod[0], count[0]
        n_unique = len(np.unique(hypnos[:, i]))
        # Deal with ties
        if n_scorers == 4:
            if count == 2 and n_unique == 2:
                # [0, 0, 1, 1] - n_unique = 2, tie, take best scorer
                # [0, 0, 1, 2] - n_unique = 3, no tie
                mod = hypnos[idx_best_hypno, i]
        elif n_scorers == 5:
            if count == 2 and n_unique == 3:
                # [0, 0, 1, 1, 2] - n_unique = 3, tie, take best scorer
                # [0, 0, 1, 2, 3] - n_unique = 4, no tie
                mod = hypnos[idx_best_hypno, i]
        elif n_scorers == 6:
            if count == 3 and n_unique == 2:
                # [0, 0, 0, 1, 1, 1]
                mod = hypnos[idx_best_hypno, i]
            if count == 2 and n_unique == 3:
                # [0, 0, 1, 1, 2, 2]
                mod = hypnos[idx_best_hypno, i]
            if count == 2 and n_unique == 4:
                # [0, 0, 1, 1, 2, 3]
                mod = hypnos[idx_best_hypno, i]
        else:
            raise ValueError("%i scorers not supported" % n_scorers)

        # If no ties, just take the mod
        hyp_cons[i] = mod
        
    return hyp_cons
