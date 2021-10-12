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
import time
import peakutils
"""
Written by:  Xingyao Wang, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn

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
    
    # RwavePos_dif = np.diff(RwavePos)
    # r_m = np.where(RwavePos_dif>=400)[0]
    # if np.size(r_m)>0:  
    #     r_add = []
    #     for fi in range(len(r_m)):
    #         y = sig[RwavePos[r_m[fi]] :RwavePos[r_m[fi]+1 ]+1]
    #         indexes = peakutils.indexes(y, thres=0.7, min_dist=70)
    #         if np.size(indexes)>0:
    #             for ri in range(len(indexes)):
    #                 r_add.append(RwavePos[r_m[fi]]+indexes[ri])
    #             # r_add.append(indexes+RwavePos[r_m[fi]])
    #         # 0.6*(y[0]+y[-1])
    #     r_add = np.array(r_add).flatten()
    #     r = np.concatenate((RwavePos,r_add), axis=-1).tolist()
    #     r.sort() 
    #     RwavePos = np.array(r)           
    # RwavePos = np.array(RwavePos)
    
    end_points = []
    rpeaks = RwavePos
    ecg_data = []    
    for i in range(len(rpeaks)-4):
        slice_i = sig[int(rpeaks[i]):int(rpeaks[i+3])]
        slice_i = StandardScaler().fit_transform(slice_i.reshape(-1,1))
        scale_ratio = 600/len(slice_i)
        tmp_data = interpolate.pchip_interpolate(np.arange(len(slice_i)) * scale_ratio, slice_i, np.arange(600))
        ecg_data.append(tmp_data)
        # print(i)        
    predict = model1.predict(np.array(ecg_data))
    tmp_pre = predict.tolist()
   # tmp_pre.insert(0,tmp_pre[0])
    tmp_pre.append(tmp_pre[-1])
    tmp_pre.append(tmp_pre[-1])
    predict1 = np.array(np.round(tmp_pre))
    
    
    sig = data[:, 0]
    ecg_data = []    
    for i in range(len(rpeaks)-4):
        slice_i = sig[int(rpeaks[i]):int(rpeaks[i+3])]
        slice_i = StandardScaler().fit_transform(slice_i.reshape(-1,1))
        scale_ratio = 600/len(slice_i)
        tmp_data = interpolate.pchip_interpolate(np.arange(len(slice_i)) * scale_ratio, slice_i, np.arange(600))
        ecg_data.append(tmp_data)
        # print(i)        
    predict = model1.predict(np.array(ecg_data))
    tmp_pre = predict.tolist()
   # tmp_pre.insert(0,tmp_pre[0])
    tmp_pre.append(tmp_pre[-1])
    tmp_pre.append(tmp_pre[-1])
    predict2 = np.array(np.round(tmp_pre))
    predict = np.logical_or(predict1, predict2)
    # predict2[np.where(np.round(predict1)>0)[0]] = 1
    # predict = np.round(predict2)
    
    beat_num = len(rpeaks)-1
    result = np.sum(np.round(predict))/beat_num
    if result < 0.1 and issubarray(np.array(np.round(predict)),np.array([1,1,1,1,1]))<5:
        test_result = 0;
    elif result > 0.9 and issubarray(np.array(np.round(predict)),np.array([0,0,0,0,0]))<5:
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
                if state_diff[j]==1 and len(np.where(np.array(np.round(predict))[j+1:j+10]==1)[0])>6 \
                    and len(np.where(np.array(np.round(predict))[0:j+1]==0)[0])>5:
                    if len(start_r)==len(end_r):
                        start_r.append(j+4)
                if state_diff[j]==-1 and len(np.where(np.array(np.round(predict))[0:j+1]==1)[0])>6 \
                    and len(np.where(np.array(np.round(predict))[j+1:j+10]==0)[0])>6:
                    if len(start_r) == 0:
                        start_r.append(0)
                        end_r.append(j+4)                       
            if j>9 and j<=len(sig)-10:
                if state_diff[j]==1 and len(np.where(np.array(np.round(predict))[j+1:j+10]==1)[0])>6 \
                    and len(np.where(np.array(np.round(predict))[j-9:j+1]==0)[0])>3:
                    if len(start_r)==len(end_r):
                        start_r.append(j+4)
                if state_diff[j]==-1 and len(np.where(np.array(np.round(predict))[j-9:j+1]==1)[0])>6 \
                    and len(np.where(np.array(np.round(predict))[j+1:j+10]==0)[0])>3:
                    if len(start_r)>len(end_r):
                        end_r.append(j+4)
            if j>len(sig)-10:
                if state_diff[j]==1 and len(np.where(np.array(np.round(predict))[j+1:]==1)[0])>5 \
                    and len(np.where(np.array(np.round(predict))[j-9:j+1]==0)[0])>6:
                    if len(start_r)==len(end_r):
                        start_r.append(j+4)
                if state_diff[j]==-1 and len(np.where(np.array(np.round(predict))[j-9:j+1]==1)[0])>6 \
                    and len(np.where(np.array(np.round(predict))[j+1:]==0)[0])>5:
                    if len(start_r)>len(end_r):
                        end_r.append(j+4)
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
    DATA_PATH = 'E:\\ECG_DATA\\CPSC2021_second_batch\\training'
    RESULT_PATH = 'E:\\CPSC_1.7643\\result22'
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    model = network.build_model()
    model.load_weights('weights01.hdf5')    
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for si, sample in enumerate(test_set):
    # for j in range(len(test_set)):
        # tic = time.time()
        # sample = test_set[j]
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path,model)
        save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)
        # print(j)
        # toc = time.time()
        # print("Elapsed time is %f sec."%(toc-tic))