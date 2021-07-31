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
    end_points = []
    rpeaks = RwavePos
    ecg_data = []
    for i in range(len(rpeaks)-4):
        start = np.round(rpeaks[i+1]-(rpeaks[i+1]-rpeaks[i])/3).astype('int')-1
        end = np.round(rpeaks[i+3]+2*(rpeaks[i+4]-rpeaks[i+3])/3).astype('int')-1
        tmp_data = data[start:end]
        # tmp_data = data[np.round(rpeaks[i+1]-(rpeaks[i+1]-rpeaks[i])/3).astype('int'):np.round(rpeaks[i+3]-(rpeaks[i+4]-rpeaks[i+3])/3).astype('int')]
        tmp_data = StandardScaler().fit_transform(tmp_data)
        tmp_data = sequence.pad_sequences(np.transpose(tmp_data), 800,dtype = 'float64',padding='pre',truncating='post')
        tmp_data = np.transpose(tmp_data)
        ecg_data.append(tmp_data)
    predict = model1.predict(np.array(ecg_data))
    beat_num = len(rpeaks)
    result = np.sum(np.round(predict))/beat_num
    if result < 0.02 and issubarray(np.array(np.round(predict)),np.array([1,1,1,1,1]))<5:
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
        if len(start_end)==1  :
            if abs(start_end[0][0]-rpeaks[0])<=30 and abs(start_end[0][-1]-rpeaks[-1])<=50 :
                start_end[0][0] = 0
                start_end[0][-1] = len(sig)-1
        end_points.extend(start_end)
        
    pred_dcit = {'predict_endpoints': end_points}
    
    return pred_dcit


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    model = network.build_model()
    model.load_weights('weights(1).hdf5')    
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path,model)

        save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)

