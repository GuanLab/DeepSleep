#!/bin/bash

set -e

i=1

dir=epoch25
mkdir -p $dir
python train_8m.py ${i} | tee -a log_${i}.txt
python single_predict_8m.py ${i}
mv auc_auprc_${i}.txt $dir
mv eva_global_${i}.txt $dir
cp weights_${i}.h5 $dir

#dir=epoch50
#mkdir -p $dir
#python continue_train_8m.py ${i} | tee -a log_${i}.txt
#python single_predict_8m.py ${i}
#mv auc_auprc_${i}.txt $dir
#mv eva_global_${i}.txt $dir
#cp weights_${i}.h5 $dir

#dir=epoch75
#mkdir -p $dir
#python continue_train1.py ${i} | tee -a log_${i}.txt
#python single_predict_8m.py ${i}
#mv auc_auprc_${i}.txt $dir
#mv eva_global_${i}.txt $dir
#cp weights_${i}.h5 $dir



