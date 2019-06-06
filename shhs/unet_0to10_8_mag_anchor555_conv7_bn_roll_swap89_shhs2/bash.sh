#!/bin/bash

set -e

for i in {1..1}
do
    dir=epoch25
    mkdir -p $dir
    python train.py ${i} | tee -a log_${i}.txt
    python single_predict.py ${i}
    mv auc_auprc_${i}.txt $dir
    mv eva_global_${i}.txt $dir
    cp weights_${i}.h5 $dir
done    
    
for i in {1..1}
do
    dir=epoch50
    mkdir -p $dir
    sed -e 's/#model.load_weights(name_model)/model.load_weights(name_model)/g' train.py > continue_train.py
    python continue_train.py ${i} | tee -a log_${i}.txt
    python single_predict.py ${i}
    mv auc_auprc_${i}.txt $dir
    mv eva_global_${i}.txt $dir
    cp weights_${i}.h5 $dir
done
    
for i in {1..1}
do
    dir=epoch75
    mkdir -p $dir
    sed -e 's/#model.load_weights(name_model)/model.load_weights(name_model)/g' train.py > continue_train.py
    python continue_train.py ${i} | tee -a log_${i}.txt
    python single_predict.py ${i}
    mv auc_auprc_${i}.txt $dir
    mv eva_global_${i}.txt $dir
    cp weights_${i}.h5 $dir
done



