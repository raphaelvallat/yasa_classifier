"""Cross-validation of the best class weights on the training data."""
import os
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GroupKFold, GridSearchCV

# Define paths
parent_dir = os.getcwd()
wdir = parent_dir + '/output/features/'
outdir = parent_dir + "/output/classifiers/"
# Load the full dataframe
df = pd.read_parquet(wdir + "features_all.parquet")

# IMPORTANT: Keep only a random sample of 20% of data
df = df.groupby(["dataset"]).sample(frac=0.2, random_state=42)
print(r"GridSearch will be performed on a random sample of 20% of data")
print("Shape after downsampling:", df.shape)

# Predictors
cols_all = df.columns
cols_time = cols_all[cols_all.str.startswith('time_')].tolist()
# EEG also includes the time columns
cols_eeg = cols_all[cols_all.str.startswith('eeg_')].tolist() + cols_time
cols_eog = cols_all[cols_all.str.startswith('eog_')].tolist()
cols_emg = cols_all[cols_all.str.startswith('emg_')].tolist()
cols_demo = ['age', 'male']

# Define predictors
X = df[cols_eeg + cols_eog + cols_emg + cols_demo].sort_index(axis=1)

# Define target and groups
y = df['stage']
subjects = df.index.get_level_values(0).to_numpy()

# Define cross-validation strategy
# For speed, we only use a 2-fold validation
cv = GroupKFold(n_splits=2)
groups = subjects

# Define hyper-parameters
# class_weight is None.
param_grid = dict(
    boosting_type=["gbdt", "dart", "goss"],
    n_estimators=[100],
    max_depth=[7, 9],
    num_leaves=[30, 50, 70],
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
    n_jobs=2, verbose=-1)
grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scorer,
                    refit=False, n_jobs=6, verbose=1)
grid.fit(X, y, groups=groups)

# Sort by best performance
cols_scoring = ["mean_test_" + c for c in scorer.keys()]
cols = ['params'] + cols_scoring

grid_res = pd.DataFrame(grid.cv_results_)[cols]
grid_res['mean_test_scores'] = grid_res[cols_scoring].mean(1)
grid_res = grid_res.sort_values(
    by="mean_test_scores", ascending=False).reset_index(drop=True).round(5)

# Export
grid_res.to_csv(outdir + "gridsearch_hyper_params.csv", index=False)
