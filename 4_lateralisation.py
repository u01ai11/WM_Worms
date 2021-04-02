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
from mpl_toolkits.mplot3d import Axes3D
try:
    import constants
    from REDTools import epoch

except:
    import sys
    sys.path.insert(0, '/home/ai05/WM_Worms')
    import constants

#%% get files
invdir = join(constants.BASE_DIRECTORY, 'inverse_ops')
fsdir = '/imaging/astle/users/ai05/RED/RED_MEG/resting/STRUCTURALS/FS_SUBDIR'
method = "MNE"
snr = 3
lambda2 = 1. / snr ** 2
epodir = join(constants.BASE_DIRECTORY, 'epoched')
files = [i for i in listdir(epodir) if 'metastim' in i]
ids = [i.split('_')[0] for i in files]

#%% just look at group average with visual evoked

# First read in the files
def get_epochs(file,dir):
    #load in epochs
    _id = file.split('_')[0]
    epochs = mne.epochs.read_epochs(join(dir, file))
    epochs.pick_types(meg=True)
    epochs.crop(tmin=-0.5, tmax=1)
    epochs.apply_baseline(baseline=(None, 0))
    #evoked = epochs.average()
    return epochs

epochs_list = joblib.Parallel(n_jobs=15)(
    joblib.delayed(get_epochs)(file,epodir) for file in files)

#%%
include = []
for i in range(len(epochs_list)):
    e = copy.deepcopy(epochs_list[i])
    evoked = e.average()
    evoked.plot_joint()
    _id = files[i].split('_')[0]
    # get inverse
    invf = [i for i in listdir(invdir) if _id in i]
    inv = mne.minimum_norm.read_inverse_operator(join(invdir, invf[0]))
    stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2,
                                         method=method, pick_ori=None)
    surfer_kwargs = dict(
        hemi='lh', subjects_dir=fsdir, views = 'lat',
        initial_time=stc.get_peak()[1], time_unit='s', size=(800, 800), smoothing_steps=5, backend='matplotlib')
    stc.plot(**surfer_kwargs)
    print("keep (1) or chuck (anything)...?")
    resp = sys.stdin.read()
    if resp.split('\n')[0] == '1':
        include.append(i)
    else:
        continue



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