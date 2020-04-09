# this is the implementation of 96 features used in this paper (2018 Dec)
# https://academic.oup.com/jamia/article/25/12/1643/5185596
# so tedious & laborious (to find how they extracted these features since they only provided descriptions without code ..)
# spend my entire day on this.. 20190221

import numpy as np
from spectrum import pmtm

def label_major_vote(input_data,scale_pool):
    size_new=int(input_data.shape[1]/scale_pool)
    input_data=input_data.reshape(size_new,scale_pool).T
    input_data=input_data.astype(int) + 1 # bincount need non-negative, int dtype
    counts=np.apply_along_axis(lambda x: np.bincount(x, minlength=3), axis=0, arr=input_data)
    major=np.apply_along_axis(lambda x: np.argmax(x), axis=0, arr=counts) - 1 
    major=major.reshape(1,len(major))
    return major

path1='/fs/project/PAS1294/osu10386/hongyang/2018/physionet/data/full_8m_anchor555/'
path2='/fs/project/PAS1294/osu10386/hongyang/2018/physionet/data/full_8m_label/'
new_path='/fs/project/PAS1294/osu10386/hongyang/2018/physionet/data/new_arousal/'
path3='/fs/project/PAS1294/osu10386/hongyang/2018/physionet/data/multitaper96/'
path4='/fs/project/PAS1294/osu10386/hongyang/2018/physionet/data/multitaper96_label/'

all_line=[]
all_ids=open('whole_train.dat','r')
for line in all_ids:
    all_line.append(line.rstrip())
all_ids.close()
all_ids=open('whole_test.dat','r')
for line in all_ids:
    all_line.append(line.rstrip())
all_ids.close()

for i in len(all_line):
    sample = all_line[i]
    the_id=sample.split('/')[-1]
    image = np.load(path1 + the_id + '.npy')[0:6,:]
    label = np.load(path2 + the_id + '.npy')
    print(the_id,i)

    i+=1
    # multitaper
    spec = [];kur=[];kurd=[];kurt=[];kura=[];kurs=[]            
    for i in np.arange(0,8384000,400): # 1310*16*400
        # default output length 512 - but it is symmetric, therefore it is 257 unique values
        # about the frequency
        # dt=1/200s, n=400 
        # df=1/(n*dt)=0.5s
        # k=5, 5 tapper -> we simply use the first one
        # we only keep the first 25 cause:
        # delta: 0.5-4Hz; theta: 4-8Hz; alpha: 8-12Hz; sigma: 12-20Hz; normalized by the total power from 0-20Hz
        sk=np.apply_along_axis(lambda x: pmtm(x,NW=3,k=5,show=False),axis=-1,arr=image[:,i:(i+400)]+1e-7) # pmtm error if input zeros
        sk=sk[:,0] # 0 is spectrum
        spec1=np.mean((abs(sk[0][0,:41]),abs(sk[1][0,:41])),axis=0)
        spec2=np.mean((abs(sk[2][0,:41]),abs(sk[3][0,:41])),axis=0)
        spec3=np.mean((abs(sk[4][0,:41]),abs(sk[5][0,:41])),axis=0)
        the_spec=np.array([spec1,spec2,spec3])
        spec.append(the_spec) # 3,41
        # kurtosis
        tmp=np.apply_along_axis(lambda x: np.sum((x-np.mean(x))**4)/400/(np.std(x)+1e-7)**4,axis=-1,arr=image[:,i:(i+400)]+1e-7) # 6,
        kur.append(tmp)
        # kurtosis - delta
        tmp=np.apply_along_axis(lambda x: np.sum((x-np.mean(x))**4)/400/(np.std(x)+1e-7)**4,axis=-1,arr=the_spec[:,1:9])
        kurd.append(tmp)
        # kurtosis - theta
        tmp=np.apply_along_axis(lambda x: np.sum((x-np.mean(x))**4)/400/(np.std(x)+1e-7)**4,axis=-1,arr=the_spec[:,9:17])
        kurt.append(tmp)
        # kurtosis - alpha
        tmp=np.apply_along_axis(lambda x: np.sum((x-np.mean(x))**4)/400/(np.std(x)+1e-7)**4,axis=-1,arr=the_spec[:,17:25])
        kura.append(tmp)                                
        # kurtosis - sigma
        tmp=np.apply_along_axis(lambda x: np.sum((x-np.mean(x))**4)/400/(np.std(x)+1e-7)**4,axis=-1,arr=the_spec[:,25:41])
        kurs.append(tmp)

    kur=np.array(kur).T # 6,16*1310
    kurd=np.array(kurd).T # 3,16*1310
    kurt=np.array(kurt).T
    kura=np.array(kura).T
    kurs=np.array(kurs).T

    spec=np.array(spec) # 20960, 3, 41
    spec=np.swapaxes(spec,0,2) # 41, 3, 20960 -> for broadcasting
    normalized_spec=spec/np.sum(spec,axis=0)

    # delta
    index=np.arange(1,9)
    delta_mean=np.apply_over_axes(np.mean,normalized_spec[index,:,:],[0]).reshape((3,20960)) # 1,3,20960 -> 3,20960
    delta_max=np.apply_over_axes(np.max,normalized_spec[index,:,:],[0]).reshape((3,20960))
    delta_min=np.apply_over_axes(np.min,normalized_spec[index,:,:],[0]).reshape((3,20960)) 
    delta_sd=np.apply_over_axes(np.std,normalized_spec[index,:,:],[0]).reshape((3,20960))

    # theta
    index=np.arange(9,17)
    theta_mean=np.apply_over_axes(np.mean,normalized_spec[index,:,:],[0]).reshape((3,20960))
    theta_max=np.apply_over_axes(np.max,normalized_spec[index,:,:],[0]).reshape((3,20960))
    theta_min=np.apply_over_axes(np.min,normalized_spec[index,:,:],[0]).reshape((3,20960))
    theta_sd=np.apply_over_axes(np.std,normalized_spec[index,:,:],[0]).reshape((3,20960))

    # alpha
    index=np.arange(17,25)
    alpha_mean=np.apply_over_axes(np.mean,normalized_spec[index,:,:],[0]).reshape((3,20960))
    alpha_max=np.apply_over_axes(np.max,normalized_spec[index,:,:],[0]).reshape((3,20960))
    alpha_min=np.apply_over_axes(np.min,normalized_spec[index,:,:],[0]).reshape((3,20960)) 
    alpha_sd=np.apply_over_axes(np.std,normalized_spec[index,:,:],[0]).reshape((3,20960))

    # delta-theta
    index=np.arange(1,9)
    dt_mean=delta_mean/theta_mean # 3,16*1310
    dt_max=delta_max/theta_max
    dt_min=delta_min/theta_min
    dt_sd=delta_sd/theta_sd

    # theta-alpha
    index=np.arange(1,9)
    ta_mean=theta_mean/alpha_mean 
    ta_max=theta_max/alpha_max
    ta_min=theta_min/alpha_min
    ta_sd=theta_sd/alpha_sd

    # delta-alpha
    index=np.arange(1,9)
    da_mean=delta_mean/alpha_mean 
    da_max=delta_max/alpha_max
    da_min=delta_min/alpha_min
    da_sd=delta_sd/alpha_sd

    line_length=abs(image[:,:8384000]-image[:,1:8384001])
    line_length=line_length.reshape((6,400,16*1310))
    line_length=np.sum(line_length,axis=1) # 6,16*1310

    image_new=np.vstack((line_length, kur, \
        delta_mean, delta_max, delta_min, delta_sd, \
        theta_mean, theta_max, theta_min, theta_sd, \
        alpha_mean, alpha_max, alpha_min, alpha_sd, \
        dt_mean, dt_max, dt_min, dt_sd, \
        ta_mean, ta_max, ta_min, ta_sd, \
        da_mean, da_max, da_min, da_sd, \
        kurd, kurt, kura, kurs)) # 96, 20960
    label_new=label_major_vote(label[:,:8384000],400)

    np.save(path3 + the_id, image_new)
    np.save(path4 + the_id, label_new)

