"""Cross-validation of the best hyper-parameters on the training data."""
import os
import optuna
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold, cross_validate

# Define paths
# parent_dir = os.path.dirname(os.getcwd())
parent_dir = "/home/walker/rvallat/yasa_classifier"  # Neurocluster
wdir = parent_dir + '/output/features/'
outdir = parent_dir + "/output/gridsearch/"

# Load the full dataframe
df = pd.read_parquet(wdir + "features_all.parquet")

# Keep only a random sample of the participants (balanced across datasets)
# idx_subj = (
#     df.reset_index()
#       .groupby('dataset')['subj']
#       .apply(lambda x: pd.Series(np.unique(x)).sample(frac=0.50, random_state=123))  # noqa
#       .to_numpy()
# )
# df = df.loc[(idx_subj), :]
# print(df.shape)

# Define predictors and target
X = df[df.columns.difference(['stage', 'dataset'])].sort_index(axis=1)
y = df['stage']
subjects = df.index.get_level_values(0).to_numpy()

# Define cross-validation strategy
cv = GroupKFold(n_splits=3)
groups = subjects


# Define objective function
def objective(trial):
    """Objective function."""
    n_est = trial.suggest_categorical(
        "n_estimators", [50, 100, 200, 300, 500])
    n_leaves = trial.suggest_int("num_leaves", 10, 110, step=10)
    max_depth = trial.suggest_int("max_depth", 3, 15, step=2)
    colsample = trial.suggest_float("colsample_bytree", 0.5, 1, step=0.1)

    clf = LGBMClassifier(
        boosting_type="gbdt", n_estimators=n_est, num_leaves=n_leaves,
        max_depth=max_depth, colsample_bytree=colsample,
        class_weight={'N1': 2, 'N2': 1, 'N3': 1, 'R': 1.4, 'W': 1.2},
        n_jobs=4, verbose=-1)

    cv_results = cross_validate(
        clf, X, y, scoring="accuracy", cv=cv, groups=groups, n_jobs=3,
        return_train_score=True)

    mean_test = cv_results['test_score'].mean()
    mean_train = cv_results['train_score'].mean()
    # https://www.jeffchiou.com/blog/hyperparameter-optimization-optuna/
    # In this case the RMSE of the difference between testing and training is
    # weighted four times less than the test accuracy.
    opt = np.sqrt((mean_test - mean_train)**2) + 4 * (1 - mean_test)
    return opt


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
print(study.best_trial)
print(study.best_params)

# Export
grid_res = study.trials_dataframe()
grid_res.drop(columns=["number", "datetime_start", "datetime_complete"],
              inplace=True)
grid_res.rename(columns={"value": "score"}, inplace=True)
grid_res.to_csv(outdir + "gridsearch_hparams_optuna.csv", index=False)
