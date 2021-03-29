"""
5_lateralisation.py
Alex Anwyl-Irvine 2021

Over the group:
    1. Check if there is a probe vs no-probe reaction

"""

from os.path import join
from os import listdir
import numpy as np
import mne
import copy
from mne.stats import linear_regression, fdr_correction
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
try:
    import constants
    from REDTools import epoch

except:
    import sys
    sys.path.insert(0, '/home/ai05/WM_Worms')
    import constants

#%% get files
epodir = join(constants.BASE_DIRECTORY, 'epoched')
check = [i for i in listdir(epodir) if 'metastim' in i]
ids = [i.split('_')[0] for i in check]

#%% try on one file
file = check[1]
epochs = mne.epochs.read_epochs(join(epodir, file))

# crop out the postcue period after pre-stim baseline
epochs.apply_baseline(baseline=(None, 0))
epochs.crop(tmin=2.0, tmax=3.5)

occ = mne.read_selection(['Left-occipital', 'Right-occipital'], info=epochs.info)
# seperate out cue directions
l = epochs['targ == 0']
r = epochs['targ == 1']

t_e = copy.deepcopy(r)
# t_e.pick_channels(occ)
#t_e.pick_types(meg='mag')
t_e.metadata = t_e.metadata.assign(Intercept=1)
names = ["Intercept", 'perc_diff']
res = linear_regression(t_e, t_e.metadata[names].reset_index(drop=True), names=names)
reject_H0, fdr_pvals = fdr_correction(res[names[1]].p_val.data)
evoked = res[names[1]].beta
evoked.plot_image(mask=reject_H0, time_unit='s')
for cond in names:
    res[cond].beta.plot_joint(title=cond, ts_args=dict(time_unit='s'),
                              topomap_args=dict(time_unit='s'))

#%% load epochs for each participant and the crop to postcue period flip left to right
def decode_probe(file):
    #load in epochs
    _id = file.split('_')[0]
    try:
        epochs = mne.epochs.read_epochs(join(epodir, file))

        if len(epochs) < 30:
            return [0]
        # crop out the postcue period after pre-stim baseline
        epochs.apply_baseline(baseline=(None, 0))
        #epochs.crop(tmin=2.0, tmax=3.5)
        epochs.metadata = epochs.metadata.assign(Intercept=1)
        epochs.pick_types(meg=True, chpi=False)

        # seperate out probe directions
        l = epochs['targ == 0']
        r = epochs['targ == 1']

        names = ["Intercept", 'perc_diff']
        res_l = linear_regression(l, l.metadata[names].reset_index(drop=True), names=names)
        res_r = linear_regression(r, r.metadata[names].reset_index(drop=True), names=names)
        return [res_l, res_r]
    except:
        return [0]

id_scores_coeff = joblib.Parallel(n_jobs=15)(
    joblib.delayed(decode_probe)(file) for file in check)

#%% calculate average coeeficiants
l_int = []; r_int = []; l_acc = []; r_acc = []

for res in id_scores_coeff:
    if res != [0]:
        l_int.append(res[0]['Intercept'].beta)
        r_int.append(res[1]['Intercept'].beta)
        l_acc.append(res[0]['perc_diff'].beta)
        r_acc.append(res[1]['perc_diff'].beta)

#%%
ldata = np.array([i.data for i in l_int])
rdata = np.array([i.data for i in r_int])


plot_l = mne.EvokedArray(ldata.mean(axis=0), l_int[0].info, tmin=2.0,
                               nave=ldata.shape[0], comment='l average')
plot_r = mne.EvokedArray(rdata.mean(axis=0), r_int[0].info, tmin=2.0,
                               nave=rdata.shape[0], comment='r average')

plot_r.plot_joint()
plot_l.plot_joint()

#mne.viz.plot_compare_evokeds([plot_l, plot_r])