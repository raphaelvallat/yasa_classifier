# qsub -pe threaded 24 -binding linear:24 -l mem_free=4G run_neurocluster.sh
python /home/walker/rvallat/yasa_classifier/gridsearch/gridsearch_hparams.py