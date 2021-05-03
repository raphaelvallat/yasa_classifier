"""Helper functions for preprocessing."""
import yasa
import numpy as np


def crop_hypno(hypno, crop=15):
    """Crop hypnogram"""
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


def extract_features(df_subj, sub, raw, include):
    """Extract features using YASA."""
    metadata = dict(age=df_subj.loc[sub, 'age'], male=df_subj.loc[sub, 'male'])
    sls = yasa.SleepStaging(
        raw,
        eeg_name=include[0],
        eog_name=include[1],
        emg_name=include[2],
        metadata=metadata)
    features = sls.get_features().reset_index()
    features['subj'] = sub
    features.set_index(['subj', 'epoch'], inplace=True)
    return features
