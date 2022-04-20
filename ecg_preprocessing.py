# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 11:43:06 2021

@author: Yating Hu
"""

import wfdb
import numpy as np
from scipy import signal,interpolate
import matplotlib.pyplot as plt
import peakutils
from biosppy.signals import ecg

def Baseline_Removal(raw_ecg1, fs):   
    baseline1 = signal.medfilt(raw_ecg1, round(0.2 * fs)+1)
    baseline2 = signal.medfilt(raw_ecg1, round(0.6 * fs)+1)   
    baseline = 0.5 * baseline1 + 0.5 * baseline2
    ecg_baseline_eliminated = raw_ecg1 - baseline
    
    return ecg_baseline_eliminated


def Burst_Removal(raw_ecg2, fs):
    ecg_burst_eliminated = raw_ecg2
    TH = 5
    s_d = np.std(raw_ecg2)
    
    TH_Low = np.mean(raw_ecg2) - TH * s_d
    TH_High = np.mean(raw_ecg2) + TH * s_d
    
    ecg_burst_eliminated[raw_ecg2 < TH_Low] = TH_Low
    ecg_burst_eliminated[raw_ecg2 > TH_High] = TH_High
    
    return ecg_burst_eliminated


def HF_Noise_Removal(raw_ecg3, fs):
    fc = 40
    filter_b,filter_a = signal.butter(3, 2 * fc / fs, 'low')   
    ecg_denoised = signal.filtfilt(filter_b, filter_a, raw_ecg3)

    return ecg_denoised


def Is_Reversed_Detection(raw_ecg4, fs):
    s_d = np.std(raw_ecg4)
    
    mecg = np.empty(len(raw_ecg4),)
    for i in range(len(raw_ecg4)):
        mecg[i] = raw_ecg4[i]
        
    mecg[abs(raw_ecg4) < 2 * s_d] = 0
    
    Pos_Energy = sum((mecg[mecg > 0]) ** 2)
    Neg_Energy = sum((mecg[mecg < 0]) ** 2)    

    isreverse = 0
    if Pos_Energy <= Neg_Energy:
        isreverse = 1
        
    return isreverse


def Pan_Tompkins_2018(ecg, fs, gr=0):
    
    if ecg is None:
        print("ecg must be a row or column vector")
    if (Pan_Tompkins_2018.__code__.co_argcount < 3):
        gr = 1
    
# initialize
    delay = 0
    skip =0                                                                    # becomes one when a T wave is detecte
    m_selected_RR = 0
    mean_RR = 0
    ser_back = 0
    ax = np. zeros(6)
    
# noise cancelation(Filtering)
    if (fs == 200):
     # remove the mean of signal
        ecg = ecg - np.mean(ecg)
     # low pass filter
        Wn = 12 * 2 / fs
        N = 3
        a,b = signal.butter(N, Wn, 'low')
        ecg_l = signal.filtfilt(a, b, ecg)
        ecg_l = ecg_l / max(abs(ecg_l))
     # start figure
        if (gr == 1 ):
           ax1 = plt.subplot(3, 2, 1)
           plt.plot(ecg)
           plt.axis('tight')
           plt.title("Raw Signal")        
           ax2 = plt.subplot(3, 2, 2)
           plt.plot(ecg_l)
           plt.axis('tight')
           plt.title("Low Pass Filtered")
           plt.show()
     # high pass filter
        Wn = 5 * 2 / fs
        N = 3
        a, b = signal.butter(N, Wn, 'high')
        ecg_h = signal.filtfilt(a, b, ecg_l)
        ecg_h = ecg_h / max(abs(ecg_h))
     # start figure
        if (gr == 1 ):        
           ax3 = plt.subplot(3, 2, 3)
           plt.plot(ecg_h)
           plt.axis('tight')
           plt.title("High Pass Filtered")
           plt.show()
           
# bandpass filter for noise cancelation of other sampling frequencies (filtering)            
    else:    
        f1 = 5                                                                 # cutoff low frequency to get rid of baseline wander
        f2 = 15                                                                # cutoff frequency to discard high frequency noise
        wn = []
        wn.append(f1 * 2 / fs)
        wn.append(f2 * 2 / fs)                                                 # cutoff based on fs
        N = 3                                                     
        a, b = signal.butter(N, wn, 'bandpass')                                                  
        ecg_h = signal.filtfilt(a, b, ecg)
        ecg_h = ecg_h / max(abs(ecg_h))
        if (gr == 1 ): 
           ax1 = plt.subplot(3, 2,(1, 2))
           plt.plot(ecg)
           plt.axis('tight')
           plt.title("Raw Signal")
           ax3 = plt.subplot(3, 2, 3)
           plt.plot(ecg_h)
           plt.axis('tight')
           plt.title("Band Pass Filtered")
           plt.show()

# derivative filter           
    if (fs != 200):
        int_c = (5 - 1) / (fs * 1 / 40)        
        x = np.arange(1,6)
        xp = np.dot(np.array([1, 2, 0, -2, -1]), (1 / 8) * fs)
        fp = np.arange(1,5+int_c,int_c)
        # b = interpolate.interp1d()        
        b = np.interp(fp, x, xp)
    else:
        b = np.dot( np.array([1, 2, 0, -2, -1]) , (1 / 8) * fs)
        
    ecg_d = signal.filtfilt(b, 1, ecg_h)
    ecg_d = ecg_d / max(ecg_d)
    
    if (gr == 1):
        ax4 = plt. subplot(3, 2, 4)
        plt.plot(ecg_d)
        plt.axis('tight')
        plt.title("Filtered with the derivative filter")
        plt.show()
        
# squaring nonlinearly enhance the dominant peaks
    # ecg_s = ecg_d ** 2
    ecg_s = np.power(ecg_d, 2)
    if (gr == 1):
        ax5 = plt. subplot(3, 2, 5)
        plt.plot(ecg_s)
        plt.axis('tight')
        plt.title("Squared")
        plt.show()
        
# moving average
    ecg_m = np.convolve( ecg_s , np.ones( int( np.around(0.150*fs) ) ) / np.around(0.150*fs) )
    delay = delay + np.around(0.150*fs) / 2
    if (gr == 1):
        ax6 = plt. subplot(3, 2, 6)
        plt.plot(ecg_m)
        plt.axis('tight')
        plt.title("Averaged with 30 samples length,Black noise,Green Adaptive Threshold,RED Sig Level,Red circles QRS adaptive threshold")
        plt.axis('tight')
        plt.show()

# fiducial marks
    locs = peakutils.indexes(ecg_m, thres=0, min_dist=np.around(0.2 * fs))
    pks = ecg_m[locs[:]]

# initialize some other parameters
    LLp = len(pks)
   # stores QRS wrt sig and filtered sig
    qrs_c = np.zeros(LLp)                                                      # amplitude of R
    qrs_i = np.zeros(LLp)                                                      # index
    qrs_i_raw = np.zeros(LLp)                                                  # amplitude of R
    qrs_amp_raw= np.zeros(LLp)                                                 # Index
   # noise buffers 
    nois_c = np.zeros(LLp)
    nois_i = np.zeros(LLp)
   # bufffers for signal and noise 
    SIGL_buf = np.zeros(LLp)
    NOISL_buf = np.zeros(LLp)
    SIGL_buf1 = np.zeros(LLp)
    NOISL_buf1 = np.zeros(LLp)
    THRS_buf1 = np.zeros(LLp)
    THRS_buf = np.zeros(LLp)

# initialize the training phase (2 seconds of the signal) to determine the THR_SIG and THR_NOISE
    THR_SIG = max(ecg_m[0:2*fs])*1/3                                           # 0.25 of the max amplitude
    THR_NOISE = np.mean(ecg_m[0:2*fs])*1/2                                     # 0.5 of the mean signal is considered to be noise
    SIG_LEV= THR_SIG
    NOISE_LEV = THR_NOISE
     
# initialize the bandpath filter threshold(2 seconds of the bandpass signal)     
    THR_SIG1 = max(ecg_h[0:2*fs])*1/3                                          # 0.25 of the max amplitude
    THR_NOISE1 = np.mean(ecg_h[0:2*fs])*1/2                                    # 0.5 of the mean signal is considered to be noise
    SIG_LEV1 = THR_SIG1                                                        # Signal level in Bandpassed filter
    NOISE_LEV1 = THR_NOISE1                                                    # Noise level in Bandpassed filter
    
# thresholding and desicion rule
    Beat_C = -1                                                                # raw beats
    Beat_C1 = -1                                                               # filtered beats
    Noise_Count = 0                                                            # noise counter
    
    for i in range(LLp):
       # locate the corresponding peak in the filtered signal
        if ((locs[i] - np.around(0.150*fs)) >= 1 and (locs[i] <= len(ecg_h))):
            _start = locs[i] - np.around(0.150*fs).astype(int)
            _ = ecg_h[_start:locs[i]]
            y_i = max(_)
            x_i = np.argmax(_)
        else:
            if i == 0:
                y_i = max(ecg_h[0:locs[i]])
                x_i = np.argmax(ecg_h[0:locs[i]])
                ser_back = 1
            elif (locs[i] >= len[ecg_h]):
                _ = ecg_h[locs[i] - np.around(0.150*fs).astype(int):]
                y_i = max(_)
                x_i = np.argmax(_)

       # update the heart_rate    
        if (Beat_C >= 9):
            diffRR = np.diff(qrs_i[Beat_C-8:Beat_C])                           # calculate RR interval 
            mean_RR = np.mean(diffRR)                                          # calculate the mean of 8 previous R waves interval
            comp = qrs_i[Beat_C] - qrs_i[Beat_C-1]                             # latest RR
            if ((comp <= 0.92*mean_RR) or (comp >= 1.16*mean_RR)):
               # lower down thresholds to detect better in MVI
                THR_SIG = 0.5*(THR_SIG)
                THR_SIG1 = 0.5*(THR_SIG1)               
            else:
                m_selected_RR = mean_RR                                        # the latest regular beats mean

       # calculate the mean last 8 R waves to ensure that QRS is not
        if m_selected_RR:
            test_m = m_selected_RR                                             # if the regular RR availabe use it 
        elif (mean_RR and m_selected_RR == 0):
            test_m = mean_RR
        else:
            test_m = 0

        if test_m:
            if ((locs[i] - qrs_i[Beat_C]) >= np.around(1.66*test_m)):          # it shows a QRS is missed
                _start = int(qrs_i[Beat_C] + np.around(0.20*fs))
                _end = int(locs[i] - np.around(0.20*fs))
                pks_temp = max(ecg_m[_start:_end+1])                           # search back and locate the max in this interval
                locs_temp = np.argmax(ecg_m[_start:_end+1])
                locs_temp = qrs_i[Beat_C] + np.around(0.20*fs) + locs_temp - 1      # location

                if (pks_temp > THR_NOISE):
                    Beat_C += 1
                    qrs_c[Beat_C] = pks_temp
                    qrs_i[Beat_C] = locs_temp
                # located in filtered sig 
                    if (locs_temp <= len(ecg_h)):
                        _start = int(locs_temp - np.around(0.150*fs))
                        _end = int(locs_temp + 1)
                        y_i_t = max(ecg_h[_start:_end])
                        x_i_t = np.argmax(ecg_h[_start:_end])
                    else:
                        _ = locs_temp - np.around(0.150*fs)
                        y_i_t = max(ecg_h[_:])
                        x_i_t = np.argmax(ecg_h[_:])

                    if (y_i_t > THR_NOISE1):
                        Beat_C1 += 1
                        qrs_i_raw[Beat_C1] = locs_temp - np.around(0.150*fs) + (x_i_t - 1)      # save index of bandpass
                        qrs_amp_raw[Beat_C1] = y_i_t                           # save amplitude of bandpass
                        SIG_LEV1 = 0.25*y_i_t + 0.75*SIG_LEV1                  # when found with the second thres

                    not_nois = 1
                    SIG_LEV = 0.25*pks_temp + 0.75*SIG_LEV       
            else:
                not_nois = 0

       # find noise and QRS peaks
        if (pks[i] >= THR_SIG): 
           #  if No QRS in 360ms of the previous QRS See if T wave 
            if (Beat_C >= 3):
                if ((locs[i] - qrs_i[Beat_C]) <= np.around(0.3600*fs)):         
                    _start = locs[i] - np.around(0.075*fs).astype('int')
                    Slope1 = np.mean(np.diff(ecg_m[_start:locs[i]]))           # mean slope of the waveform at that position
                    _start = int(qrs_i[Beat_C] - np.around(0.075*fs))
                    _end = int(qrs_i[Beat_C])
                    Slope2 = np.mean(np.diff(ecg_m[_start:_end]))              # mean slope of previous R wave
                    if abs(Slope1) <= abs(0.5*(Slope2)):                       # slope less then 0.5 of previous R
                        nois_c[Noise_Count] = pks[i]
                        nois_i[Noise_Count] = locs[i]
                        Noise_Count += 1
                        skip = 1                                               # T wave identification
                       # adjust noise level
                        NOISE_LEV1 = 0.125*y_i + 0.875*NOISE_LEV1
                        NOISE_LEV = 0.125*pks[i] + 0.875*NOISE_LEV
                    else:
                        skip = 0

            if (skip == 0):                                                    # skip is 1 when a T wave is detected
                Beat_C += 1
                qrs_c[Beat_C] = pks[i]
                qrs_i[Beat_C] = locs[i]

                if (y_i >= THR_SIG1):
                    Beat_C1 += 1
                    if ser_back:
                        qrs_i_raw[Beat_C1] = x_i
                    else:
                        qrs_i_raw[Beat_C1] = locs[i] - np.around(0.150*fs) + (x_i - 1)

                    qrs_amp_raw[Beat_C1] =  y_i
                    SIG_LEV1 = 0.125*y_i + 0.875*SIG_LEV1

                SIG_LEV = 0.125*pks[i] + 0.875*SIG_LEV

        elif ((THR_NOISE <= pks[i]) and (pks[i] < THR_SIG)):
            NOISE_LEV1 = 0.125*y_i + 0.875*NOISE_LEV1
            NOISE_LEV = 0.125*pks[i] + 0.875*NOISE_LEV     
        elif (pks[i] < THR_NOISE):
            nois_c[Noise_Count] = pks[i]
            nois_i[Noise_Count] = locs[i]    
            NOISE_LEV1 = 0.125*y_i + 0.875*NOISE_LEV1    
            NOISE_LEV = 0.125*pks[i] + 0.875*NOISE_LEV
            Noise_Count += 1

        # Adjust the threshold with SNR
        if (NOISE_LEV != 0 or SIG_LEV != 0):
            THR_SIG = NOISE_LEV + 0.25*(abs(SIG_LEV - NOISE_LEV))
            THR_NOISE = 0.5*(THR_SIG)

        if (NOISE_LEV1 != 0 or SIG_LEV1 != 0):
            THR_SIG1 = NOISE_LEV1 + 0.25*(abs(SIG_LEV1 - NOISE_LEV1))
            THR_NOISE1 = 0.5*(THR_SIG1)

        SIGL_buf[i] = SIG_LEV
        NOISL_buf[i] = NOISE_LEV
        THRS_buf[i] = THR_SIG

        SIGL_buf1[i] = SIG_LEV1
        NOISL_buf1[i] = NOISE_LEV1
        THRS_buf1[i] = THR_SIG1

        skip = 0                                                  
        not_nois = 0
        ser_back = 0

    # Adjust lengths
    qrs_i_raw = qrs_i_raw[0:Beat_C1+1]
    qrs_amp_raw = qrs_amp_raw[0:Beat_C1+1]
    qrs_c = qrs_c[0:Beat_C+1]
    qrs_i = qrs_i[0:Beat_C+1]

    return qrs_i_raw   #, qrs_amp_raw

def Rwave_Detection(ecg, fs):
    # ecg=ecg[:] 
    
   # 检测信号是否需要翻转
   # fl = Is_Reversed_Detection(ecg,fs)
    # if (Is_Reversed_Detection(ecg,fs)==1):
    #     ecg = np.negative(ecg)
    f1 = 12                                                                 # cutoff low frequency to get rid of baseline wander
    f2 = 35                                                                # cutoff frequency to discard high frequency noise
    wn = []
    wn.append(f1 * 2 / fs)
    wn.append(f2 * 2 / fs)
    filter_b,filter_a = signal.butter(3, wn, 'bandpass')   
    
    ecg = signal.filtfilt(filter_b, filter_a, ecg)            
    qrs_i_raw = Pan_Tompkins_2018(ecg,fs)
    qrs_i_raw=np.array(qrs_i_raw,dtype="int32")
    
    Win_Len = np.floor(0.05*fs)
    flag = np.ones(len(qrs_i_raw))
    tem_RwavePos = qrs_i_raw.copy()
    
    
    for kr in range(len(qrs_i_raw)):
        if (qrs_i_raw[kr] - Win_Len <= 0) or (qrs_i_raw[kr] + Win_Len > len(ecg)):
            flag[kr] = 0
        else:
            _start = int(qrs_i_raw[kr] - Win_Len)
            _end = int( qrs_i_raw[kr] + Win_Len)
            ma_pos = np.argmax(ecg[ _start: _end+1])
            tem_RwavePos[kr] = qrs_i_raw[kr] + ma_pos - Win_Len
            
    idx = np.where(flag==1)
    RwavePos = tem_RwavePos.copy()
    RwavePos[idx] = tem_RwavePos[idx]
    
    return RwavePos







def load_data(sample_path):                              
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']
    # print('signal:',sig)
    # print('fields:',fields)
    return sig, fields, length, fs


def preprocess(sample_path):
    sig, fields, length, fs = load_data(sample_path)    
    # sqi_search_step=round(1*fs)
    # Signal_Quality_Win_Len=5*fs
    # QRS_Pos_Err=0.10;   
    # Signal_Quality_TH=0.85; 
    
    raw_ecg1 = sig[:, 1]
    # sqi_flag = signal_quality_est_ver2(raw_ecg1, fs, QRS_Pos_Err, Signal_Quality_Win_Len, sqi_search_step, Signal_Quality_TH)
        
    ecg_baseline_eliminated = Baseline_Removal(raw_ecg1, fs)
    ecg_burst_eliminated = Burst_Removal(ecg_baseline_eliminated, fs)
    ecg_HF_denoised = HF_Noise_Removal(ecg_burst_eliminated, fs)

    RwavePos = Rwave_Detection(ecg_HF_denoised, fs=200)
    sig[:,1] = ecg_HF_denoised
    raw_ecg1 = sig[:, 0]
    ecg_baseline_eliminated = Baseline_Removal(raw_ecg1, fs)
    ecg_burst_eliminated = Burst_Removal(ecg_baseline_eliminated, fs)
    ecg_HF_denoised = HF_Noise_Removal(ecg_burst_eliminated, fs)
    sig[:,0] = ecg_HF_denoised
    # cycle_start=[]
    # cycle_end=[]
    # for cn in range(len(RwavePos)-2):
    #     tcycle_start=RwavePos[cn]
    #     tcycle_end=RwavePos[cn+1]-1
    #     if (sqi_flag[tcycle_start:tcycle_end+1]==1).all():
    #         cycle_start.append(tcycle_start)
    #         cycle_end.append(tcycle_end)
    
    return sig,length,RwavePos

# recordname = "0_1"
# sample_path = "E:/ECG_DATA/CPSC2021_second_batch/training_I/data_"+recordname
# sig,length,RwavePos,cycle_start,cycle_end = preprocess(sample_path)
