# qsub -pe threaded 20 -binding linear:20 -l mem_free=4G run_neurocluster.sh
python /home/walker/rvallat/yasa_classifier/gridsearch/gridsearch_smoothing_scaling.py