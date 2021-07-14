"""Cross-validation of the best class weights on the training data."""
import os
import pandas as pd
from itertools import product
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, f1_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold, GridSearchCV

labels = ["N1", "N2", "N3", "R", "W"]


# Define paths
parent_dir = os.path.dirname(os.getcwd())
# parent_dir = "/home/walker/rvallat/yasa_classifier"  # Neurocluster
wdir = parent_dir + '/output/features/'
outdir = parent_dir + "/output/gridsearch/"
# Load the full dataframe
df = pd.read_parquet(wdir + "features_all.parquet")

# IMPORTANT: Keep only a random subset of each dataset
# df = df.groupby(["dataset"]).sample(frac=0.50, random_state=42)
# print(r"GridSearch will be performed on a random sample of 50% of data")
# print("Shape after downsampling:", df.shape)

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

# Show the values of balanced class weights
# print("Balanced class weights are:",
#       np.round(compute_class_weight('balanced', np.unique(y), y), 2))

# Define cross-validation strategy
cv = GroupKFold(n_splits=3)
groups = subjects

# Define hyper-parameters
params = dict(
    boosting_type='gbdt',
    n_estimators=50,
    max_depth=7,
    num_leaves=30,
    colsample_bytree=0.8,
    importance_type='gain',
    n_jobs=5
)

# Define scoring metrics
scorer = {
    "accuracy": "accuracy",
    "f1_N1": make_scorer(f1_score, labels=["N1"], average=None),
    "f1_N2": make_scorer(f1_score, labels=["N2"], average=None),
    "f1_N3": make_scorer(f1_score, labels=["N3"], average=None),
    "f1_R": make_scorer(f1_score, labels=["R"], average=None),
    "f1_W": make_scorer(f1_score, labels=["W"], average=None),
    "mcc": make_scorer(matthews_corrcoef),
}

# Create parameter grid
weights = {
    "N1": [1.6, 1.8, 2, 2.2],
    "N2": [0.8, 1],
    "N3": [1, 1.2, 1.4],
    "R": [1, 1.2, 1.4],
    "W": [1, 1.2, 1.4],
}

# For testing
# weights = dict.fromkeys(labels, [1])

param_grid = []
prods = list(product(*weights.values()))
print(f"There are {len(prods)} unique combinations of weights.")

for w in prods:
    p = {}
    p['class_weight'] = [
        {'N1': w[0], 'N2': w[1], 'N3': w[2], 'R': w[3], 'W': w[4]}]
    param_grid.append(p)

# Fit GridSearchCV
clf = LGBMClassifier(**params)
grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scorer,
                    refit=False, n_jobs=3, verbose=3)
grid.fit(X, y, groups=groups)

# Sort by best performance
cols_scoring = ["mean_test_" + c for c in scorer.keys()]
cols = ['param_class_weight'] + cols_scoring
grid_res = pd.DataFrame(grid.cv_results_)[cols]

grid_res.rename(
    columns={'param_class_weight': 'class_weight'}, inplace=True)

grid_res['mean_test_scores'] = grid_res[cols_scoring].mean(1)
grid_res = grid_res.sort_values(
    by="mean_test_scores", ascending=False).reset_index(drop=True).round(5)

# Export to CSV
grid_res.to_csv(outdir + "gridsearch_class_weights.csv", index=False)
