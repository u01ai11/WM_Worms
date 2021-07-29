"""
6_behavioural.py
Alex Anwyl-Irvine 2021

1. Characterise error for cued vs uncued
2. Does error/cue usage predict IQ scores
3. Does error/cue usage predict WM measures
4. SES effects

"""
from os.path import join
from os import listdir
import numpy as np
import mne
import scipy.stats as ss
from statsmodels.stats import anova
from statsmodels.stats import weightstats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import (SlidingEstimator, cross_val_multiscore)
import joblib
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import copy
try:
    import constants
    from REDTools import epoch

except:
    import sys
    sys.path.insert(0, '/home/ai05/WM_Worms')
    import constants

import os

#%%
file_includes = np.load(join(constants.BASE_DIRECTORY, 'good_visual_evoked.npy'))
good_ids = [i.split('_')[0] for i in file_includes]
meta = pd.read_csv(join(constants.BASE_DIRECTORY, 'Combined3.csv'))
all_trials = pd.read_csv(join(constants.BASE_DIRECTORY, 'all_trials.csv'))
all_trials['offset'] = all_trials['targ_ang'] - all_trials['resp_ang']
all_trials['offset'] = all_trials['offset'].astype('float')
all_trials['offset_abs'] = all_trials['offset'].abs()
all_trials['cue_type'] = ['neutral' if i == -1 else 'valid' for i in all_trials['cue_dir']]

def ang_dist(f, zero):
    f = np.array(f)
    o_t = f[:,0]
    o_r = f[:,1]
    # moduli after division
    ori1 = o_t % zero
    ori2 = o_r % zero

    # calculate the difference
    error = ori2 - ori1
    # where the difference is larger then a clockwise 90 degree rotation, it
    # should be counted as a counter-clockwise (negative) rotation
    error[error > zero / 2] = -1 * (zero - error[error > zero / 2])
    error[error <= -zero / 2] = (zero + error[error <= -zero / 2])
    return error

all_trials['ang_dist'] = ang_dist(all_trials[['targ_ang', 'resp_ang']],90)

all_trials.to_csv(join(constants.BASE_DIRECTORY, 'all_trials.csv'))
#%%
"""
1. ERROR FOR CUED AND UNCUED TRIALS 
"""

d

#1.1 Calculate participantwise Accuracy and Precision
#Accuract: reciprocal of the circulat standard deviation
for i in range(good_ids):
    _id = int(good_ids[0])
    _trials = all_trials[all_trials.id ==_id]
    _trials = _trials[_trials.prac == 0]

    _acc_v = 1- ss.circstd(_trials[_trials.cue_type == 'valid'].offset)
    _acc_n = 1- ss.circstd(_trials[_trials.cue_type == 'neutral'].offset)

