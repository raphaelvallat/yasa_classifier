"""Cross-validation of the best scaling method on the training data."""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import robust_scale, scale
from sklearn.metrics import make_scorer, f1_score

labels = ["N1", "N2", "N3", "R", "W"]

# Define paths
parent_dir = os.path.dirname(os.getcwd())
wdir = parent_dir + '/output/features/'
outdir = parent_dir + "/output/gridsearch/"

# Load the full dataframe
df = pd.read_parquet(wdir + "features_all.parquet")

# Keep only a random sample of the participants (balanced across datasets)
idx_subj = (
    df.reset_index()
      .groupby('dataset')['subj']
      .apply(lambda x: pd.Series(np.unique(x)).sample(frac=0.50, random_state=42))  # noqa
      .to_numpy()
)
df = df.loc[(idx_subj), :]
print(df.shape)

# Columns that will be normalized
cols_to_norm = df.columns[
    (df.columns.str.startswith("e")) & (~df.columns.str.contains("norm"))]

# Define temporal smoothing windows
p, c = 'p2', 'c7'

# For speed, we only use a 3-fold validation
subjects = df.index.get_level_values(0).to_numpy()
cv = GroupKFold(n_splits=3)
groups = subjects

print("There are %i nights in cross-validation dataset (%i folds)"
      % (len(np.unique(subjects)), cv.get_n_splits()))

# Define hyper-parameters (including class weights)
params = dict(
    boosting_type='gbdt',
    n_estimators=50,
    max_depth=7,
    num_leaves=30,
    class_weight={'N1': 2.2, 'N2': 1, 'N3': 1, 'R': 1.2, 'W': 1},
    colsample_bytree=0.8,
    importance_type='gain',
    n_jobs=4
)

print(params)
clf = LGBMClassifier(**params)

# Define scoring metrics
scorer = {
    "accuracy": "accuracy",
    "f1_N1": make_scorer(f1_score, labels=["N1"], average=None),
    "f1_N2": make_scorer(f1_score, labels=["N2"], average=None),
    "f1_N3": make_scorer(f1_score, labels=["N3"], average=None),
    "f1_R": make_scorer(f1_score, labels=["R"], average=None),
    "f1_W": make_scorer(f1_score, labels=["W"], average=None),
}

# Initialize output dict
grid_res = {
    "scale": [],
    "accuracy": [],
    "f1_N1": [],
    "f1_N2": [],
    "f1_N3": [],
    "f1_R": [],
    "f1_W": [],
}


def rscale5(x):
    """5-95% robust scaler."""
    return robust_scale(x, quantile_range=(5, 95))


def rscale10(x):
    """10-90% robust scaler."""
    return robust_scale(x, quantile_range=(10, 90))


def rscale25(x):
    """25-75% robust scaler."""
    return robust_scale(x, quantile_range=(25, 75))


# Define GroupBy object and index
grp = df.groupby(level=0, sort=False)[cols_to_norm]
epochs = df.index.get_level_values(1)

# Loop across scaling
for sc in ['r5', 'r10', 'r25', 'standard']:
    if sc == "r5":
        func = rscale5
    elif sc == "r10":
        func = rscale10
    elif sc == "r25":
        func = rscale25
    else:
        func = scale

    # Print info
    now = datetime.now()
    print("%s | %s" % (now, sc))

    # Remove all the columns with norm
    cols_norm = df.columns[df.columns.str.contains("norm")]
    df.drop(columns=cols_norm, inplace=True)

    # Apply the rolling windows on each night separately
    # .. past
    w = 2 * int(p[1:])
    rollp = grp.rolling(window=w, min_periods=1).mean()
    rollp = rollp.groupby(level=0).transform(func).astype(np.float32)
    rollp = rollp.add_suffix('_%smin_norm' % p)
    rollp['epoch'] = epochs
    rollp.set_index("epoch", append=True, inplace=True)
    df = df.join(rollp)

    # .. centered
    w = 2 * int(c[1:]) + 1  # Add one to get symmetrical window
    rollc = grp.rolling(
        window=w, center=True, min_periods=1, win_type='triang').mean()
    rollc = rollc.groupby(level=0).transform(func).astype(np.float32)
    rollc = rollc.add_suffix('_%smin_norm' % c)
    rollc['epoch'] = epochs
    rollc.set_index("epoch", append=True, inplace=True)
    df = df.join(rollc)

    # Cross-validate
    X = df[df.columns.difference(['stage', 'dataset'])]
    y = df['stage']
    scores = cross_validate(
        clf, X, y, cv=cv, groups=groups, scoring=scorer, n_jobs=3)

    # Append to main output dict
    grid_res['scale'].append(sc)
    grid_res['accuracy'].append(scores['test_accuracy'].mean())
    grid_res['f1_N1'].append(scores['test_f1_N1'].mean())
    grid_res['f1_N2'].append(scores['test_f1_N2'].mean())
    grid_res['f1_N3'].append(scores['test_f1_N3'].mean())
    grid_res['f1_R'].append(scores['test_f1_R'].mean())
    grid_res['f1_W'].append(scores['test_f1_W'].mean())

# Convert to a dataframe
grid_res = pd.DataFrame(grid_res)

# Calculate the mean test scores
grid_res['mean_test_scores'] = grid_res.iloc[:, 1:].mean(1)
grid_res.sort_values(by=['mean_test_scores'], ascending=False, inplace=True)

# Export to CSV
grid_res.to_csv(outdir + "gridsearch_scaling.csv", index=False)
