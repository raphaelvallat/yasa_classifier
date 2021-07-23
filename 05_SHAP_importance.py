"""SHAP features importance."""
import os
import shap
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

# Define paths
# parent_dir = os.getcwd()
parent_dir = "/home/walker/rvallat/yasa_classifier"  # Neurocluster
wdir = parent_dir + '/output/features/'
outdir = parent_dir + "/output/classifiers/"
assert os.path.isdir(wdir)
assert os.path.isdir(outdir)

# Load the full dataframe
df = pd.read_parquet(wdir + "features_all.parquet")

# Define predictors and target
X = df[df.columns.difference(['stage', 'dataset'])].sort_index(axis=1)
y = df['stage']

print(df.shape)
print(X.columns)

# Define hyper-parameters
params = dict(
    boosting_type='gbdt',
    n_estimators=400,
    max_depth=5,
    num_leaves=90,
    colsample_bytree=0.5,
    importance_type='gain',
    n_jobs=20
)

# Manually define class weight
# See output/classifiers/gridsearch_class_weights.xlsx
params['class_weight'] = {'N1': 2.2, 'N2': 1, 'N3': 1.2, 'R': 1.4, 'W': 1}
fname = outdir + 'clf_eeg+eog+emg+demo_lgb_gbdt_custom_shap'

# Fit classifier
clf = LGBMClassifier(**params)
clf.fit(X, y)

print("Fitting done!")

# Calculate SHAP feature importance - we limit the number of trees for speed
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X, tree_limit=50)
# Sum absolute values across all stages and then average across all samples
shap_sum = np.abs(shap_values).sum(axis=0).mean(axis=0)
df_shap = pd.Series(shap_sum, index=X.columns.tolist(), name="Importance")
df_shap.sort_values(ascending=False, inplace=True)
df_shap.index.name = 'Features'

# Export
np.savez_compressed(fname + ".npz", shap_values=shap_values)
df_shap.to_csv(fname + ".csv")

# Disabled: plot
# from matplotlib import colors
# cmap_stages = ['#99d7f1', '#009DDC', 'xkcd:twilight blue',
#                'xkcd:rich purple', 'xkcd:sunflower']
# cmap = colors.ListedColormap(np.array(cmap_stages)[class_inds])
# class_inds = np.argsort(
#   [-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
# shap.summary_plot(shap_values, X, plot_type='bar', max_display=15,
#                   color=cmap, class_names=clf.classes_)