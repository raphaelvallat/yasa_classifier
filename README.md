# yasa_classifiers

This repository contains the notebooks used to generate the pre-trained classifier of YASA's sleep staging module:
https://raphaelvallat.com/yasa/build/html/generated/yasa.SleepStaging.html

The datasets can be found on sleepdata.org. You need to request data access to download the datasets. Specifically, training of the sleep staging classifier is done using the following datasets: CCSHS, CFS, CHAT, HomePAP, MESA, MrOS, SHHS.

## Steps

 1. `01_features_nsrr_\*.ipynb`: calculate features from the raw PSG files. Make sure to update the paths!
 2. `02_create_classifiers.ipynb`: train and export the sleep staging classifier.
