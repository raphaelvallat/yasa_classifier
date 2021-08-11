# yasa_classifiers

This repository contains the notebooks used to generate the pre-trained classifier of [YASA sleep staging module](https://raphaelvallat.com/yasa/build/html/generated/yasa.SleepStaging.html). The main GitHub repository of YASA can be found [here](https://github.com/raphaelvallat/yasa).

For more details on the algorithm and its validation, please refer to the [preprint article](https://www.biorxiv.org/content/10.1101/2021.05.28.446165v1.abstract).

The datasets can be found on [sleepdata.org](sleepdata.org). You need to request data access to download the datasets. Specifically, training of the sleep staging classifier is done using the following datasets: CCSHS, CFS, CHAT, HomePAP, MESA, MrOS, SHHS.

If you have questions, please contact Dr. Raphael Vallat (<raphaelvallat@berkeley.edu>).

## Steps

To reproduce all the results in the [preprint article](https://www.biorxiv.org/content/10.1101/2021.05.28.446165v1.abstract), you need to run the following scripts/notebooks in order:

0. `00_randomize_train_test.ipynb`: randomize NSRR nights to training / testing set. This assumes that you have previously downloaded the NSRR data on your computer (up to 4 TB). 
1. `feature_extraction/01_features_*.ipynb`: calculate the features for all the training nights.
2. `02_create_classifiers.ipynb`: train and export the sleep staging classifier.
3. `predict/03_predict_*.ipynb`: apply the algorithm on testing nights.
4. `04_validation_*.ipynb`: evaluate performance on the testing sets (testing set 1 = NSRR, testing set 2 = DREEM).
5. `05_SHAP_importance.py`: calculate the SHAP features importance on the NSRR training set
6. `06_nsrr_demographics.ipynb`: compare the demographics and health data of the NSRR training / testing set.

In addition, the scripts in the `gridsearch` folder perform parameter searchs with cross-validation to find the best hyper-parameters, class weights and temporal smoothing windows.
