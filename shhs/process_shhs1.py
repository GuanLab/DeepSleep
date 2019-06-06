## parse annotation
import xml.etree.ElementTree as ET
import mne
import numpy as np
import sys
import os
import scipy.io
import cv2

def anchor (ref, ori): # input m*n np array
    d0=ori.shape[0]
    d1=ori.shape[1]
    ref=cv2.resize(ref,(d1,d0),interpolation=cv2.INTER_AREA)
    ori_new=ori.copy()
    for i in range(d0):
        ori_new[i,np.argsort(ori[i,:])]=ref[i,:]
    return ori_new

ref555=np.load('/ssd/hongyang/2018/physionet/data/ref555.npy')[np.array([2,3,6,7,10,11,12]),:]
ref555=cv2.resize(ref555,(int(5550000/8),7),interpolation=cv2.INTER_AREA)

path1='/ssd/hongyang/2018/physionet/data/shhs/'
path2='/ssd/hongyang/2018/physionet/data/shhs_image/'
path3='/ssd/hongyang/2018/physionet/data/shhs_label/'
os.system('mkdir -p ' + path2)
os.system('mkdir -p ' + path3)

size= 2**20
scale_pool=5
freq=125

# test records
id_all=[]
f=open('id1.txt','r')
for line in f:
    id_all.append(line.rstrip())
f.close()

num=0
for the_id in id_all:
    the_id=the_id.rstrip()
    print(the_id,num)
    num+=1
    # image
    edf = mne.io.read_raw_edf(path1 + the_id + '.edf', verbose=False)
    image_ori = edf.get_data()[np.array([2,7,5,4,8,0,3]),:]
    image_ori = cv2.resize(image_ori,(int(edf.n_times/scale_pool),7),interpolation=cv2.INTER_AREA)
    image = anchor(ref555, image_ori)
    d0=image.shape[0]
    d1=image.shape[1]
    if(d1 < size):
        image=np.concatenate((image,np.zeros((d0,size-d1))),axis=1)
    np.save(path2 + the_id , image)
    # label
    label=np.zeros(d1)
    root = ET.parse(path1 + the_id + '-nsrr.xml').getroot()
    for i in np.arange(1,len(root[2])):
        if root[2][i][0].text == 'Arousals|Arousals':
            start=int(float(root[2][i][2].text) * float(freq) / float(scale_pool))
            end=start + int(float(root[2][i][3].text) * float(freq) / float(scale_pool))
            label[start:end]=1
    print(np.sum(label)/float(len(label)))
    np.save(path3 + the_id , label)


