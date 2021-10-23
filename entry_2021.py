#!/usr/bin/env python3

import numpy as np
import os
import sys

import wfdb
from utils import qrs_detect, comp_cosEn, save_dict
from ecg_preprocessing import preprocess
import network
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from scipy import signal,interpolate
"""
Written by:  
    
Save answers to '.json' files, the format is as {‘predict_endpoints’: [[s0, e0], [s1, e1], …, [sm-1, em-2]]}.
"""

def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']

    return sig, length, fs

def ngrams_rr(data, length):
    grams = []
    for i in range(0, length-12, 12):
        grams.append(data[i: i+12])
    return grams

def issubarray(array,a):
    count = 0
    for i in range(len(array)-len(a)):
        if (a==array[i:i+len(a)]).all():
            count+=1

    return count

def challenge_entry(sample_path,model1):
    """
    This is a baseline method.
    """
    data,length,RwavePos = preprocess(sample_path)
    sig = data[:, 1]
    
    RwavePos_dif = np.diff(RwavePos)
    r_f = np.where(RwavePos_dif<100)[0]
    if np.size(r_f)>0:  
        for fi in range(len(r_f)):
            # print(fi)
            if r_f[fi]>0 and r_f[fi]<(len(RwavePos)-2):                            
                if sig[RwavePos[r_f[fi]]] < 0.6*sig[RwavePos[r_f[fi]+1]] or sig[RwavePos[r_f[fi]]] < 0.6*sig[RwavePos[r_f[fi]-1]] :
                    RwavePos[r_f[fi]] = 0
                if sig[RwavePos[r_f[fi]+1]] < 0.6*sig[RwavePos[r_f[fi]]] or sig[RwavePos[r_f[fi]+1]] < 0.6*sig[RwavePos[r_f[fi]+2]] :
                    RwavePos[r_f[fi]+1] = 0
            elif r_f[fi]==(len(RwavePos)-2):                            
                if sig[RwavePos[r_f[fi]]] < 0.6*(sig[RwavePos[r_f[fi]-1]] + sig[RwavePos[r_f[fi]-2]] ):  
                    RwavePos[r_f[fi]] = 0                
            elif r_f[fi]==0:                            
                if sig[RwavePos[r_f[fi]]] < 0.6*(sig[RwavePos[r_f[fi]+1]] + sig[RwavePos[r_f[fi]+2]] ):
                    RwavePos[r_f[fi]] = 0                          
    RwavePos = RwavePos[np.where(RwavePos>0)[0]] 
       
    RwavePos = np.array(RwavePos)
    end_points = []
    rpeaks = RwavePos
    ecg_data = []    
    for i in range(len(rpeaks)-4):
        slice_i = sig[rpeaks[i]:rpeaks[i+3]]
        slice_i = StandardScaler().fit_transform(slice_i.reshape(-1,1))
        scale_ratio = 600/len(slice_i)
        tmp_data = interpolate.pchip_interpolate(np.arange(len(slice_i)) * scale_ratio, slice_i, np.arange(600))
        ecg_data.append(tmp_data)
        
    predict = model1.predict(np.array(ecg_data))
    tmp_pre = predict.tolist()
   # tmp_pre.insert(0,tmp_pre[0])
    tmp_pre.append(tmp_pre[-1])
    tmp_pre.append(tmp_pre[-1])
    predict = np.array(tmp_pre)
    beat_num = len(rpeaks)-1
    result = np.sum(np.round(predict))/beat_num
    if result < 0.1 and issubarray(np.array(np.round(predict)),np.array([1,1,1,1,1]))<7:
        test_result = 0;
    elif result > 0.9 and issubarray(np.array(np.round(predict)),np.array([0,0,0,0,0]))<7:
        test_result = 1;
    else:
        test_result = 2;    
    end_points = []
    if test_result == 1:
        end_points.append([0, len(sig)-1])
    elif test_result == 2:
        state_diff = np.diff(np.round(predict).transpose())[0]
        start_r = []
        end_r = []
        for j in range(len(state_diff)):
            if j-9 <0:
                if state_diff[j]==1 and len(np.where(np.array(np.round(predict))[j+1:j+10]==1)[0])>=5 \
                    and len(np.where(np.array(np.round(predict))[0:j+1]==0)[0])>=5:
                    if len(start_r)==len(end_r):
                        # start_r.append(j+4)
                        start_r.append(j+3)
                if state_diff[j]==-1 and len(np.where(np.array(np.round(predict))[0:j+1]==1)[0])>=5 \
                    and len(np.where(np.array(np.round(predict))[j+1:j+10]==0)[0])>=5:
                    if len(start_r) == 0:
                        start_r.append(0)
                        # end_r.append(j+4)       
                        end_r.append(j+3)                        
            if j>9 and j<=len(sig)-10:
                if state_diff[j]==1 and len(np.where(np.array(np.round(predict))[j+1:j+10]==1)[0])>=5 \
                    and len(np.where(np.array(np.round(predict))[j-9:j+1]==0)[0])>=5:
                    if len(start_r)==len(end_r):
                        # start_r.append(j+4)
                        start_r.append(j+3)
                if state_diff[j]==-1 and len(np.where(np.array(np.round(predict))[j-9:j+1]==1)[0])>=5 \
                    and len(np.where(np.array(np.round(predict))[j+1:j+10]==0)[0])>=5:
                    if len(start_r)>len(end_r):
                        # end_r.append(j+4)      
                        end_r.append(j+3)
            if j>len(sig)-10:
                if state_diff[j]==1 and len(np.where(np.array(np.round(predict))[j+1:]==1)[0])>=5 \
                    and len(np.where(np.array(np.round(predict))[j-9:j+1]==0)[0])>=5:
                    if len(start_r)==len(end_r):
                        # start_r.append(j+4)
                        start_r.append(j+3)
                if state_diff[j]==-1 and len(np.where(np.array(np.round(predict))[j-9:j+1]==1)[0])>=5 \
                    and len(np.where(np.array(np.round(predict))[j+1:]==0)[0])>=5:
                    if len(start_r)>len(end_r):
                        # end_r.append(j+4)      
                        end_r.append(j+3)
        if len(start_r)==0:
            start_r.append(0)
        if len(start_r)>len(end_r):
            end_r.append(-1)
        start_r = np.expand_dims(start_r, -1)
        end_r = np.expand_dims(end_r, -1)
        start_end = np.concatenate((rpeaks[start_r], rpeaks[end_r]-1), axis=-1).tolist()      
        end_points.extend(start_end)
        
    pred_dict = {'predict_endpoints': end_points}
    
    return pred_dict


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
# 
    # DATA_PATH = 'E:\\ECG_DATA\\CPSC2021_second_batch\\training'
    # RESULT_PATH = 'E:\\1.7691\\result3'
    
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    model = network.build_model()
    model.load_weights('weights01.hdf5')    
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
    # for i in range(len(test_set)):
        # sample = test_set[i]    
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path,model)
        save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)

