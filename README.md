# yasa_classifiers

This repository contains the notebooks used to generate the pre-trained classifier of YASA's sleep staging module:
https://raphaelvallat.com/yasa/build/html/generated/yasa.SleepStaging.html

The datasets can be found on sleepdata.org. You need to request data access to download the datasets. Specifically, training of the sleep staging classifier is done using the following datasets: CCSHS, CFS, CHAT, HomePAP, MESA, MrOS, SHHS.

If you have questions, please contact Dr. Raphael Vallat <raphaelvallat9@gmail.com>.

## Steps

0. `00_randomize_train_test.ipynb`: randomize NSRR nights to training / testing.
1. `feature_extraction/01_features_*.ipynb`: calculate PSG features for all the training nights.
2. `02_create_classifiers.ipynb`: train and export the sleep staging classifier.
3. `predict/03_predict_*.ipynb`: apply the algorithm on testing nights.
4. `04_validation_*.ipynb`: evaluate performance on the testing sets (testing set 1 = NSRR, testing set 2 = DREEM).

In addition, the `gridsearch_hyperparams.py` and `gridsearch_weights.py` scripts perform a grid search with cross validation on the model's hyperparameters and class weights, respectively.
