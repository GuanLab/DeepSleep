#!/usr/bin/env python
import os
import sys
import numpy as np
import scipy.io

id_all=[]
f=open('id1.txt','r')
for line in f:
    id_all.append(line.rstrip())
f.close()

# shuffle
id_all=np.array(id_all[:1000])
np.random.seed(449)
index=np.arange(len(id_all))
np.random.shuffle(index)
id_all=id_all[index]

f=open('id_train1.dat', 'w')
for the_id in id_all[:750]:
    f.write('%s\n' % the_id)
f.close()

f=open('id_test1.dat', 'w')
for the_id in id_all[750:]:
    f.write('%s\n' % the_id)
f.close()


id_all=[]
f=open('id2.txt','r')
for line in f:
    id_all.append(line.rstrip())
f.close()

# shuffle
id_all=np.array(id_all[:1000])
np.random.seed(449)
index=np.arange(len(id_all))
np.random.shuffle(index)
id_all=id_all[index]

f=open('id_train2.dat', 'w')
for the_id in id_all[:750]:
    f.write('%s\n' % the_id)
f.close()

f=open('id_test2.dat', 'w')
for the_id in id_all[750:]:
    f.write('%s\n' % the_id)
f.close()














