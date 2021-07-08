"""Cross-validation of the number of estimators on the training data."""
import os
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GroupKFold, GridSearchCV

# Define paths
parent_dir = os.path.dirname(os.getcwd())
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
param_grid = dict(
    boosting_type=["gbdt"],  # ["gbdt", "dart", "goss"],
    n_estimators=[10, 20, 30, 50, 75, 100, 200, 300, 500, 750, 1000],
    max_depth=[7],
    num_leaves=[70],
    colsample_bytree=[0.8],
)

# Define scoring metrics
scorer = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "mcc": make_scorer(matthews_corrcoef),
}

# Fit GridSearchCV
clf = LGBMClassifier(
    class_weight={'N1': 2, 'N2': 1, 'N3': 1, 'R': 1.4, 'W': 1.2},
    n_jobs=4, verbose=-1)
grid = GridSearchCV(
    clf, param_grid, cv=cv, scoring=scorer, refit=False, n_jobs=3,
    return_train_score=True, verbose=1)
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
grid_res.to_csv(outdir + "gridsearch_hparams_nestimators.csv", index=False)
