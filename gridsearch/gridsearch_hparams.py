"""Cross-validation of the hyper-parameters on the training data."""
import os
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GroupKFold, GridSearchCV

# Define paths
# parent_dir = os.path.dirname(os.getcwd())
parent_dir = "/home/walker/rvallat/yasa_classifier"  # Neurocluster
wdir = parent_dir + '/output/features/'
outdir = parent_dir + "/output/gridsearch/"

# Load the full dataframe
df = pd.read_parquet(wdir + "features_all.parquet")

# Define predictors
X = df[df.columns.difference(['stage', 'dataset'])].sort_index(axis=1)

# Define target and groups
y = df['stage']
subjects = df.index.get_level_values(0).to_numpy()

# Define cross-validation strategy
cv = GroupKFold(n_splits=3)
groups = subjects

# Define hyper-parameters
# .. Run 1
# param_grid = dict(
#     boosting_type=["gbdt"],  # ["gbdt", "dart", "goss"],
#     n_estimators=[50, 100, 300, 500],
#     max_depth=[5, 7, 9],
#     num_leaves=[30, 50, 70, 90],
#     colsample_bytree=[0.6, 0.8],
# )

# .. Run 2
# param_grid = dict(
#     boosting_type=["gbdt", "dart"],
#     n_estimators=[200, 400],
#     max_depth=[5, 7],
#     num_leaves=[30, 90],
#     colsample_bytree=[0.5, 0.7],
# )

# .. Run 3
param_grid = dict(
    boosting_type=["gbdt"],
    n_estimators=[400],
    max_depth=[4, 6, 8],
    num_leaves=[40, 80],
    colsample_bytree=[0.5, 0.6],
)

# Define scoring metrics
scorer = {
    "accuracy": "accuracy",
    "f1_N1": make_scorer(f1_score, labels=["N1"], average=None),
    "f1_N2": make_scorer(f1_score, labels=["N2"], average=None),
    "f1_N3": make_scorer(f1_score, labels=["N3"], average=None),
    "f1_R": make_scorer(f1_score, labels=["R"], average=None),
    "f1_W": make_scorer(f1_score, labels=["W"], average=None),
}

# Fit GridSearchCV
clf = LGBMClassifier(
    n_jobs=6, verbose=-1)
grid = GridSearchCV(
    clf, param_grid, cv=cv, scoring=scorer, refit=False, n_jobs=3,
    return_train_score=True, verbose=3)
grid.fit(X, y, groups=groups)

# Sort by best performance
cols_test = ["mean_test_" + c for c in scorer.keys()]
cols_train = ["mean_train_" + c for c in scorer.keys()]
cols = ['params'] + cols_test + cols_train

grid_res = pd.DataFrame(grid.cv_results_)[cols]
grid_res['mean_test_scores'] = grid_res[cols_test].mean(1)
grid_res = grid_res.sort_values(
    by="mean_test_scores", ascending=False).reset_index(drop=True).round(5)

# Export
grid_res.to_csv(outdir + "gridsearch_hparams.csv", index=False)
