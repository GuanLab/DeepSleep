#!/bin/bash

set -e

#for i in {3,2,4,5,1}

for i in {2..5}
do
    sed -e "s/i=1/i=${i}/g" < run1.pbs > run${i}.pbs
done

for i in {1..5}
do
    qsub run${i}.pbs
done




