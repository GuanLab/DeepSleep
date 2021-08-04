import numpy as np
import scipy.io
import cv2
import os

def import_signals(file_name): # feature
    return scipy.io.loadmat(file_name)['val']

def import_arousals(file_name): # target
    import h5py
    import numpy as np
    f = h5py.File(file_name, 'r')
    arousals = np.transpose(np.array(f['data']['arousals']))
    return arousals

def label_major_vote(input_data,scale_pool):
    size_new=int(input_data.shape[1]/scale_pool)
    input_data=input_data.reshape(size_new,scale_pool).T
    input_data=input_data.astype(int) + 1 # bincount need non-negative, int dtype
    counts=np.apply_along_axis(lambda x: np.bincount(x, minlength=3), axis=0, arr=input_data)
    major=np.apply_along_axis(lambda x: np.argmax(x), axis=0, arr=counts) - 1 
    major=major.reshape(1,len(major))
    return major

path1='./data/training/'
path2='./data/feature_8m/'
path3='./data/label_8m/'

os.mkdir(path2)
os.mkdir(path3)

size= 2**23
num_pool=0
scale_pool=2**num_pool
num=0

#all_ids=os.listdir(path1)
all_ids=open('./data/list_id.txt','r')
for each_id in all_ids:
    each_id=each_id.rstrip()
    print(each_id,num)
    num+=1
    arousal_file = path1 + each_id + '/' + each_id + '-arousal.mat'
    label = import_arousals(arousal_file)
    signal_file = path1 + each_id + '/' + each_id + '.mat'
    image = import_signals(signal_file)
    d0=image.shape[0]
    d1=image.shape[1]
    if(d1 < size*scale_pool):
        image=np.concatenate((image,np.zeros((d0,size*scale_pool-d1))),axis=1)
        label=np.concatenate((label,np.zeros((1,size*scale_pool-d1))-1),axis=1)
#    image=cv2.resize(image,(size,d0),interpolation=cv2.INTER_AREA) # average pool
#    label=label_major_vote(label,scale_pool)
    np.save(path2 + each_id , image)
    np.save(path3 + each_id , label)
all_ids.close()

