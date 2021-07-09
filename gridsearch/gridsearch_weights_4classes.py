"""Cross-validation of the best class weights on the training data."""
import os
import pandas as pd
from itertools import product
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, f1_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold, GridSearchCV


def get_param_grid():
    """Create combination of weights."""
    weights = {
        "LIGHT": [0.8, 0.9, 1],
        "DEEP": [1, 1.2, 1.4],
        "REM": [1, 1.2, 1.4],
        "WAKE": [1, 1.2, 1.4],
    }

    # For testing
    # labels = ["LIGHT", "DEEP", "REM", "WAKE"]
    # weights = dict.fromkeys(labels, [1])

    param_grid = []
    prods = list(product(*weights.values()))
    print(f"There are {len(prods)} unique combinations of weights.")

    for w in prods:
        p = {}
        p['class_weight'] = [
            {'LIGHT': w[0], 'DEEP': w[1], 'REM': w[2], 'WAKE': w[3]}]
        param_grid.append(p)
    return param_grid


def main():
    """Main."""

    # Define paths
    parent_dir = os.path.dirname(os.getcwd())
    wdir = parent_dir + '/output/features/'
    outdir = parent_dir + "/output/gridsearch/"
    # Load the full dataframe
    df = pd.read_parquet(wdir + "features_all.parquet")

    # Replace 5 classes to 4 classes
    df['stage'].replace({
        'N1': 'LIGHT', 'N2': 'LIGHT', 'N3': 'DEEP', 'R': 'REM', 'W': 'WAKE'},
        inplace=True)

    print(df['stage'].value_counts(normalize=True))

    # IMPORTANT: Keep only a random sample of 25% of data
    df = df.groupby(["dataset"]).sample(frac=0.25, random_state=42)
    print(r"GridSearch will be performed on a random sample of 25% of data")
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

    # Show the values of balanced class weights
    # print("Balanced class weights are:",
    #       np.round(compute_class_weight('balanced', np.unique(y), y), 2))

    # Define cross-validation strategy
    # For speed, we only use a 2-fold validation
    cv = GroupKFold(n_splits=2)
    groups = subjects

    # Define hyper-parameters
    params = dict(
        boosting_type='gbdt',
        n_estimators=50,
        max_depth=7,
        num_leaves=30,
        colsample_bytree=0.8,
        importance_type='gain',
        n_jobs=2
    )

    # Define scoring metrics
    scorer = {
        "accuracy": "accuracy",
        "f1_LIGHT": make_scorer(f1_score, labels=["LIGHT"], average=None),
        "f1_DEEP": make_scorer(f1_score, labels=["DEEP"], average=None),
        "f1_REM": make_scorer(f1_score, labels=["REM"], average=None),
        "f1_WAKE": make_scorer(f1_score, labels=["WAKE"], average=None),
        "mcc": make_scorer(matthews_corrcoef),
    }

    # get param_grid
    param_grid = get_param_grid()

    # Fit GridSearchCV
    clf = LGBMClassifier(**params)
    grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scorer,
                        refit=False, n_jobs=6, verbose=3)
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
    grid_res.to_csv(outdir + "gridsearch_class_weights_4classes.csv",
                    index=False)


if __name__ == "__main__":
    main()
