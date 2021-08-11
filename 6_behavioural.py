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

#%% load in ALL data (including bad IDs)
datadir = join(constants.BASE_DIRECTORY, 'behav')
data_files = listdir(datadir)
data_trials = [i for i in data_files if 'trials' in i]
df = pd.read_csv(join(datadir, data_trials[1]), delimiter="\t")
all_trials = pd.DataFrame(columns=list(df.columns) + ['id'])
ids = [i.split('_')[0] for i in data_trials]
for _id in ids:
    _file = [f for f in data_trials if _id in f]
    print(len(_file))
    if len(_file) == 0:
        continue
    try:
        _df = pd.read_csv(join(datadir, _file[0]), delimiter="\t")
    except:
        continue
    _idcol = pd.DataFrame({'id': [_id]*len(_df)})
    _df = _df.join(_idcol)
    all_trials = all_trials.append(_df, ignore_index=True)
    good_ids.append(_id)
#%%
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

#all_trials.to_csv(join(constants.BASE_DIRECTORY, 'all_trials.csv'))

#%% load meta
meta = pd.read_csv(join(constants.BASE_DIRECTORY, 'Combined3.csv'))

#%%
"""
1. ERROR FOR CUED AND UNCUED TRIALS 
"""

def precision(_list):
    return 1/ss.circstd(np.deg2rad(_list), low=0, high= np.deg2rad(180))

#1.1 Calculate participantwise Accuracy and Precision
#Accuracy: reciprocal of the circulat standard deviation - i.e. vpn misses
group_r = []
for _id in all_trials.id.unique():
    print(_id)
    _trials = all_trials[all_trials.id ==_id]
    _trials = _trials[_trials.prac == 0]
    _trials = _trials.iloc[0:60]
    _prec_all = precision(_trials.offset.abs())
    _prec_v = precision(_trials[_trials.cue_type == 'valid'].offset.abs().to_list())
    _prec_n = precision(_trials[_trials.cue_type == 'neutral'].offset.abs().to_list())
    _prec_l = precision(_trials[_trials.cue_dir == 0].offset.abs().to_list())
    _prec_r = precision(_trials[_trials.cue_dir == 1].offset.abs().to_list())

    _acc_all = np.mean(np.deg2rad(_trials.ang_dist.abs().to_list()))
    _acc_v = np.deg2rad(np.mean(np.abs(_trials[_trials.cue_type == 'valid'].ang_dist)))
    _acc_n = np.deg2rad(np.mean(np.abs(_trials[_trials.cue_type == 'neutral'].ang_dist)))
    _acc_l = np.deg2rad(np.mean(np.abs(_trials[_trials.cue_dir == 0].ang_dist)))
    _acc_r = np.deg2rad(np.mean(np.abs(_trials[_trials.cue_dir == 1].ang_dist)))
    group_r.append((_acc_v, _acc_n, _acc_l, _acc_r, _acc_all, _prec_v, _prec_n, _prec_l, _prec_r, _prec_all, float(_id)))

#%% create combined spreadsheet

comb = pd.DataFrame(data=np.array(group_r), columns=['Error Valid', 'Error Neutral', 'Error Left', 'Error Right', 'Overall Error', 'Precision Valid',
                                                     'Precision Neutral', 'Precision Left', 'Precision Right', 'Overall Precision', 'id'])
comb['id'] = [str(int(i)) for i in comb['id']]
comb['Alex_ID'] = [str(i) for i in comb['id']]
meta['Alex_ID'] = [str(i) for i in meta['Alex_ID']]
comb = comb.merge(meta, on='Alex_ID')

comb.to_csv(join(constants.BASE_DIRECTORY, 'worms_meta.csv'))

#%%
comb = pd.read_csv(join(constants.BASE_DIRECTORY, 'worms_meta.csv'))
#%%
plt.close('all')
sns.regplot(data=comb, x='objective_SES', y='Overall Error')
plt.show()
#%% interp

group_r = np.array(pd.DataFrame(group_r).fillna(pd.DataFrame(group_r).mean()))
#%% error over time (i.e. learning)
group_t = np.zeros([len(good_ids), 12,4])
windowsize = 10
for i, _id in enumerate(good_ids):
    _trials = all_trials[all_trials.id ==_id]
    _trials = _trials[_trials.prac == 0]
    # _trials['block'] = [i for i in range(int(len(_trials)/10)) for _ in range(10)]
    # _mean_perf = np.mean(np.array(_trials.ang_dist.abs()).reshape(-1, 10), axis=1)

    #_mean_v = np.mean(np.array(_trials[_trials.cue_type == 'valid'].ang_dist.abs()).reshape(-1, 6), axis=1)
    #_mean_n = np.mean(np.array(_trials[_trials.cue_type == 'neutral'].ang_dist.abs()).reshape(-1, 4), axis=1)
    #cue_benefit = _mean_v - _mean_n
    plt.close('all')
    plt.scatter(y=_trials.ang_dist.abs(), x=list(range(len(_trials.ang_dist.abs()))))
    plt.show()
    #group_t[i,:,:] = np.array([_mean_perf, _mean_v, _mean_n, cue_benefit])
#%%
ss.stats.ttest_ind(a=group_r[:,3], b=group_r[:,4])
#%% Is there an effect of cue?
plt.close('all')
tmp = pd.DataFrame(group_r[:,3:5], columns=['prec_v', 'prec_n'])
tmp =pd.melt(tmp, value_vars=["prec_v", "prec_n"])
sns.histplot(data=tmp, x="value", hue="variable", kde=True)
plt.show()
 #%%
sns.displot(data=None, x=group_r[:,5], kde=True)
plt.show()


#%% First look at precision and error across all Paritcipants
plt.close('all')
sns.set_style("whitegrid")
#sns.set(font_scale=1.5)
tmp = pd.melt(comb, value_vars=["Overall Error", "Overall Precision"])
g = sns.FacetGrid(tmp, col="variable", sharex=False, sharey=True, despine=True, height=6)
g.map(sns.histplot, "value", common_norm=True, kde=True)
g.map(sns.rugplot, "value")
g.set_axis_labels(y_var='Count')
g.axes[0][0].set_xlabel('Error (Radians)')
g.axes[0][1].set_xlabel('Precision (1/SD)')
g.set_titles(col_template="", row_template="")
plt.tight_layout()


#%% Left vs Right vs Nutral Cue
tmp = pd.melt(comb, value_vars=["Error Neutral", "Error Left", "Error Right"], id_vars=['id'])
fig, ax = plt.subplots(1,2)
sns.boxplot(x="variable", y="value", data=tmp, ax=ax[0])
tmp = pd.melt(comb, value_vars=["Precision Neutral", "Precision Left", "Precision Right"], id_vars=['id'])
sns.boxplot(x="variable", y="value", data=tmp, ax=ax[1])
#%% Age and Error - valid vs neutral cues
covariate = "objective_SES"
fig, ax = plt.subplots(1,2)
sns.regplot(data=comb, y="Error Neutral", x=covariate, label="Neutral Cue", ax=ax[0])
sns.regplot(data=comb, y="Error Valid", x=covariate, label="Valid Cue",ax=ax[0])
ax[0].set_ylabel('Error (Radians)')
sns.regplot(data=comb, y="Precision Neutral", x=covariate, ax=ax[1])
sns.regplot(data=comb, y="Precision Valid", x=covariate,ax=ax[1])
ax[1].set_ylabel('Precision (1/SD)')
fig.legend()
plt.tight_layout()
