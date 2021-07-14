"""Cross-validation of the best class weights on the training data."""
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
    w_N1 = trial.suggest_float("N1", 1, 3, step=0.2)
    w_N2 = trial.suggest_float("N2", 0.6, 1.4, step=0.2)
    w_N3 = trial.suggest_float("N3", 1, 1.6, step=0.2)
    w_R = trial.suggest_float("R", 1, 1.6, step=0.2)
    w_W = trial.suggest_float("W", 1, 1.6, step=0.2)

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
        boosting_type="gbdt", n_estimators=50, num_leaves=30,
        max_depth=7, colsample_bytree=0.8,
        class_weight={'N1': w_N1, 'N2': w_N2, 'N3': w_N3, 'R': w_R, 'W': w_W},
        n_jobs=4, verbose=-1)

    cv_results = cross_validate(
        clf, X, y, scoring=scorer, cv=cv, groups=groups, n_jobs=3,
        return_train_score=False)

    mean_test = []
    for c in scorer.keys():
        mean_test.append(cv_results["test_%s" % c].mean())

    mean_test = np.mean(mean_test)
    return mean_test


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
print(study.best_trial)
print(study.best_params)

# Export
grid_res = study.trials_dataframe()
grid_res.drop(columns=["number", "datetime_start", "datetime_complete"],
              inplace=True)
grid_res.rename(columns={"value": "score"}, inplace=True)
grid_res.to_csv(outdir + "gridsearch_weights_optuna.csv", index=False)
