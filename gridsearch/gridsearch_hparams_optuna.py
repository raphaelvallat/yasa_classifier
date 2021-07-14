"""Cross-validation of the best hyper-parameters on the training data."""
import os
import optuna
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GroupKFold, cross_validate

# Define paths
# parent_dir = os.path.dirname(os.getcwd())
parent_dir = "/home/walker/rvallat/yasa_classifier"  # Neurocluster
wdir = parent_dir + '/output/features/'
outdir = parent_dir + "/output/gridsearch/"

# Load the full dataframe
df = pd.read_parquet(wdir + "features_all.parquet")

# OPTIONAL: Keep only a random sample of participants (balanced across dataset)
# idx_subj = (
#     df.reset_index()
#       .groupby('dataset')['subj']
#       .apply(lambda x: pd.Series(np.unique(x)).sample(frac=0.01, random_state=123))  # noqa
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
    # Define parameter space
    n_est = trial.suggest_int("n_estimators", 50, 500, step=50)
    n_leaves = trial.suggest_int("num_leaves", 10, 110, step=10)
    max_depth = trial.suggest_int("max_depth", 3, 15, step=2)
    colsample = trial.suggest_float("colsample_bytree", 0.5, 1, step=0.1)

    # Define scorer
    scorer = {
        "accuracy": "accuracy",
        "f1_N1": make_scorer(f1_score, labels=["N1"], average=None),
        "f1_N2": make_scorer(f1_score, labels=["N2"], average=None),
        "f1_N3": make_scorer(f1_score, labels=["N3"], average=None),
        "f1_R": make_scorer(f1_score, labels=["R"], average=None),
        "f1_W": make_scorer(f1_score, labels=["W"], average=None),
    }

    # Create estimator and cross-validate
    clf = LGBMClassifier(
        boosting_type="gbdt", n_estimators=n_est, num_leaves=n_leaves,
        max_depth=max_depth, colsample_bytree=colsample,
        n_jobs=6, verbose=-1)

    cv_results = cross_validate(
        clf, X, y, scoring=scorer, cv=cv, groups=groups, n_jobs=3,
        return_train_score=True)

    print(cv_results)

    mean_test, mean_train = [], []
    for c in scorer.keys():
        mean_test.append(cv_results["test_%s" % c].mean())
        mean_train.append(cv_results["train_%s" % c].mean())

    mean_test = np.mean(mean_test)
    mean_train = np.mean(mean_train)
    # https://www.jeffchiou.com/blog/hyperparameter-optimization-optuna/
    opt = np.abs(mean_train - mean_test) + 4 * (1 - mean_test)
    return opt


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)
print(study.best_trial)
print(study.best_params)

# Export
grid_res = study.trials_dataframe()
grid_res.drop(columns=["number", "datetime_start", "datetime_complete"],
              inplace=True)
grid_res.rename(columns={"value": "score"}, inplace=True)
grid_res.to_csv(outdir + "gridsearch_hparams_optuna.csv", index=False)
