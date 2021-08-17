"""
4_cue_analysis.py
Alex Anwyl-Irvine 2021

Over the group:
    1. Load in data, filter epochs and calculated evoked objects
    2. Check if there is a valid vs neutral cue effect on evoked data
    3. Check if there is a time-frequency component to this effect
    4. Check if this effect is predicted by performance
    5. Check if this effect is predicted by other covariates

"""

from os.path import join
from os import listdir
import numpy as np
import mne
from functools import partial
from mne.stats import ttest_1samp_no_p
import copy
from mne.stats import linear_regression, fdr_correction
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glmtools
import sails
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
import statsmodels.api as sm
import scipy.stats as ss
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

epodir = join(constants.BASE_DIRECTORY, 'epoched_final')
files = [i for i in listdir(epodir) if 'metastim' in i]
ids = [i.split('_')[0] for i in files]

#%Select files with 'good' visual evoked responses
file_includes = np.load(join(constants.BASE_DIRECTORY, 'good_visual_evoked.npy'))
#file_includes = files

#%% just make an average evoked period

"""
1. Loading, filtering and Evoked
"""
"""
TIMINGS FOR REFERENCE
0 - STIMULUS: duration 1
1 - MAINTENANCE: duration 1
2 - CUE: duration 0.5
2.5 - POSTCUE: duration 1
3.5 - PROBE
"""

# Let's just use magnetometers
sen = 'mag'

event_labels = ['Stimulus', 'Cue', 'Probe']
event_onsets = [0, 2, 3.5]
event_offsets = [1, 2.5, 4.496]
bad_thresh = 10

def remove_bad(epochs, bad_thresh):
    bad_es = []
    for i in range(len(epochs)):
        bad_es.append(sails.utils.detect_artefacts(epochs[i]._data[0], axis=1,
                                                   reject_mode='dim',
                                                   ret_mode='bad_inds').sum())
    bad_es_mask = np.array(bad_es) > bad_thresh
    epochs = epochs[~bad_es_mask]
    return epochs

def av_evoked(file):
    try:
        epochs = mne.epochs.read_epochs(join(epodir, file))
        epochs.pick_types(meg=sen)
        epochs
        epochs.apply_baseline(baseline=(-0.5, 0))
        #epochs.crop(tmin=2.5, tmax=3.5)

        epochs = remove_bad(epochs, bad_thresh)

        epochs = epochs.equalize_event_counts(['cue_dir == 0', 'cue_dir == 1', 'cue_dir == -1'])[0]
        l = epochs['cue_dir == 0']
        r = epochs['cue_dir == 1']
        n = epochs['cue_dir == -1']
        cued = epochs['cue_dir == 0 or cue_dir == 1']
        #mne.epochs.equalize_epoch_counts([n, cued])
        lav = l.average()
        rav = r.average()
        nav = n.average()
        cav = cued.average()

        l_cue = mne.combine_evoked([lav, nav], [1,-1])
        r_cue = mne.combine_evoked([rav, nav], [1, -1])
        cue = mne.combine_evoked([cav, nav], [1, -1])
        _all= epochs.average()
    except:
        print(file)
        print(epochs)
    #return(lav,rav)
    return[(lav, rav, nav, _all, cav), (l, r, n, epochs)]

    #return(l_cue, r_cue)

l_r = joblib.Parallel(n_jobs=15)(
    joblib.delayed(av_evoked)(file) for file in file_includes)

#%%
np.save(join(constants.BASE_DIRECTORY, 'cue_erp_data_grad.npy'), l_r, allow_pickle=True)
#%%
l_r_ep = [i[1] for i in l_r]
l_r = [i[0] for i in l_r]

#%%
l_r = np.load(join(constants.BASE_DIRECTORY, 'cue_erp_data.npy'), allow_pickle=True)
l_r_ep = [i[1] for i in l_r]
l_r = [i[0] for i in l_r]
#%% Inspect
counts = [len(i[3]) for i in l_r_ep]

#%% sample average
#all_sample = mne.combine_evoked([i[3] for i in l_r], weights=[1]*len(l_r))
all_sample = mne.combine_evoked([i[3] for i in l_r], weights='nave')
#%% cluster test
X_L = np.empty((len(l_r), len(l_r[0][0].times), l_r[0][0].data.shape[0]))
X_R = np.empty((len(l_r), len(l_r[0][0].times), l_r[0][0].data.shape[0]))
X_N = np.empty((len(l_r), len(l_r[0][0].times), l_r[0][0].data.shape[0]))
X_A  = np.empty((len(l_r), len(l_r[0][0].times), l_r[0][0].data.shape[0]))
X_C =  np.empty((len(l_r), len(l_r[0][0].times), l_r[0][0].data.shape[0]))
L_E = []
R_E = []
N_E = []
A_E = []
C_E = []
for i in range(len(l_r)):
    X_L[i,:,:] = np.transpose(l_r[i][0].data, (1,0))
    X_R[i,:,:] = np.transpose(l_r[i][1].data, (1,0))
    X_N[i, :, :] = np.transpose(l_r[i][2].data, (1, 0))
    X_A[i, :, :] = np.transpose(l_r[i][3].data, (1, 0))
    X_C[i, :, :] = np.transpose(l_r[i][4].data, (1, 0))
    L_E.append(l_r[i][0])
    R_E.append(l_r[i][1])
    N_E.append(l_r[i][2])
    A_E.append(l_r[i][3])
    C_E.append(l_r[i][4])
#connectivity strutctrue
adjacency = mne.channels.find_ch_adjacency(l_r[0][0].info, ch_type=sen)
#%%
"""
2. Do we have a difference between cue types (neutral and valid)
"""
data = (X_C - X_N)
crop_time = [1,3.5]
#crop_time = [1, 4]
threshold = 4  # very high, but the test is quite sensitive on this data
#threshold = dict(start=4., step=0.1)
# set family-wise p-value
p_accept = 0.05
sigma = 1e-3
stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
#crop data
pre_zero = 0.5
sfreq = 250
data = data[:,int((crop_time[0]+pre_zero)*sfreq):int((crop_time[1]+pre_zero)*sfreq),:]

cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test(data, n_permutations=1000,
                                             threshold=threshold, tail=0,
                                             n_jobs=13, buffer_size=None,
                                             adjacency=adjacency[0], stat_fun=stat_fun_hat)
# data = [X_C[:,int((crop_time[0]+pre_zero)*sfreq):int((crop_time[1]+pre_zero)*sfreq),:],
#        X_N[:,int((crop_time[0]+pre_zero)*sfreq):int((crop_time[1]+pre_zero)*sfreq),:]]
# threshold = 18
# cluster_stats = mne.stats.spatio_temporal_cluster_test(data, n_permutations=1000,
#                                              threshold=threshold, tail=0,
#                                              n_jobs=13, buffer_size=None,
#                                              adjacency=adjacency[0])

#np.save(join(constants.BASE_DIRECTORY,'erp_cue_clusterstats.npy'), cluster_stats, allow_pickle=True)
#%
#cluster_stats = np.load(join(constants.BASE_DIRECTORY,'erp_cue_clusterstats.npy'), allow_pickle=True)
T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]
print(good_cluster_inds)
print(p_values.min())
print([np.unique(clusters[i][0]) for i in good_cluster_inds])
#%% Plot clusters
sns.set(style="whitegrid", font_scale=1.5, palette="Set2")
labels_plot = ['Valid Cue', 'Neutral Cue']
evs = copy.copy([C_E, N_E])
for elist in evs:
    for ev in elist:
        ev.crop(crop_time[0], crop_time[1])
colors = {labels_plot[0]: "crimson", labels_plot[1]: 'steelblue'}
linestyles = {"L": '-', "R": '--'}

# get sensor positions via layout
pos = mne.find_layout(evs[0][0].info).pos

picks, pos2, merge_channels, names, ch_type, sphere, clip_origin = \
    mne.viz.topomap._prepare_topomap_plot(evs[0][0].info, sen)

reg_masks = []

plot_t_offset = 2
# loop over clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    f_map = T_obs[time_inds, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = evs[0][0].times[time_inds]

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True
    reg_masks.append(mask)
    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(20, 6))

    # plot average test statistic and mark significant sensors
    image, _ = mne.viz.plot_topomap(f_map, pos, mask=mask, axes=ax_topo, ch_type=sen,
                            vmin=np.min, vmax=np.max, sphere=None, show=False)

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'T Stastics ({:0.2f} - {:0.2f} s)'.format(*sig_times[[0, -1]]))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.6)
    title = 'Retro-Cue Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"
    mne.viz.plot_compare_evokeds({labels_plot[0]:evs[0], labels_plot[1]:evs[1]}, title=title, picks=ch_inds, axes=ax_signals,
                         #colors=colors,
                         show=False,
                         split_legend=True, truncate_yaxis='auto', combine='mean')

    ax_signals.set_title(title, y=1.1)
    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='red', alpha=0.3)

    xmin, xmax = ax_signals.get_xlim()

    # add labels if needed
    for i, ev in enumerate(event_labels):
        if xmin <= event_onsets[i] <= xmax:
            ax_signals.axvline(event_onsets[i], linestyle = '--',color='black')
            ax_signals.text(event_onsets[i], ymax, f'{ev} onset')
        if xmin <= event_offsets[i] <= xmax:
            ax_signals.axvline(event_offsets[i], linestyle = '--',color='black')
            ax_signals.text(event_offsets[i], ymax, f'{ev} offset')

    ax_signals.set_ylabel('Field Strength (fT)')
    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)

    labels = [item.get_text() for item in ax_signals.get_xticklabels()]
    labels = [str(float(i)-plot_t_offset) for i in labels]
    #ax_signals.set_xticklabels(labels)
    plt.tight_layout()
    plt.show()


#%% when we mask this off, can we use it to predict pefrmance?

#for each participants mask off the difference
time_inds, space_inds = np.squeeze(clusters[2])
mask_array = np.zeros(data.shape[1:3], dtype=int)
for i in range(data.shape[1]):
    for ii in range(data.shape[2]):
        if (i in time_inds) & (ii in space_inds):
            mask_array[i,ii] =1
mask_array = mask_array.astype(bool)

vals = []
for i in range(data.shape[0]):
    vals.append(data[i][mask_array].mean())

#get ids
matching_ids = [i.split('_')[0] for i in file_includes]
# load in all meta data
meta = pd.read_csv(join(constants.BASE_DIRECTORY, 'worms_meta.csv'))
meta['id'] = meta.id.astype(str)
meta = meta[meta.Alex_ID.isin(matching_ids)]

#reorder to match meta
vals_reord = []
for _id in meta.id:
    tmp_ind = matching_ids.index(_id)
    vals_reord.append(vals[tmp_ind])

meta['Cluster Summed'] = vals_reord
meta.to_csv(join(constants.BASE_DIRECTORY, 'worms_cluster_meta.csv'))
#%% OLS regression on the performance variables

def permute(mod, perms):
    nulls = []
    for i in range(mod.data.exog.shape[1]):
        this_null = []
        for ii in range(perms):
            pmod = copy.deepcopy(mod)
            pmod.data.exog[:,i] = np.random.permutation(mod.data.exog[:,i])
            this_null.append(pmod.fit().tvalues[i])
        nulls.append(np.array(this_null))
    return np.array(nulls)

#prepare data
x = np.array(meta['Cluster Summed']) # to be predicted
y = np.array(meta[['Overall Error', 'Overall Precision', 'Overall Onset', 'Overall Duration']]) # predictors
y = ss.zscore(y)
#y = np.array(meta[['Overall Error', 'Overall Precision']]).mean(axis=1)
y = sm.add_constant(y, prepend=False) # intercept/constant
mod = sm.OLS(x, y, hasconst=True)
res = mod.fit()
print(res.summary())

nulls = permute(mod,5000)
reals = res.tvalues
monte_p = [ss.percentileofscore(_nulls, t) for _nulls, t in zip(nulls, reals)]

print()
#%%


#%% d
"""
4. What does this look like in the time-frequency domain?
"""


# Let's just use magnetometers
sen = 'grad'
event_labels = ['Stimulus', 'Cue', 'Probe']
event_onsets = [0, 2, 3.5]
event_offsets = [1, 2.5, 4.496]
bad_thresh = 10
sfreq = 250
crop_time = [1,3.5]
base_line = [-0.5,0]
def time_freq(file):
    epochs = mne.epochs.read_epochs(join(epodir, file))
    epochs.pick_types(meg=sen)
    epochs = remove_bad(epochs, bad_thresh)
    epochs = epochs.equalize_event_counts(['cue_dir == 0', 'cue_dir == 1', 'cue_dir == -1'])[0]
    decim = 4
    freqs = np.arange(5, 40, 1)  # define frequencies of interest
    tfr_epochs = mne.time_frequency.tfr_morlet(epochs, freqs, n_cycles=freqs/2,
                            average=False, return_itc=False, decim=decim, n_jobs=2)
    # tfr_epochs = mne.time_frequency.tfr_multitaper(epochs, freqs, n_cycles=freqs/2, decim=decim,
    #                         average=False, return_itc=False, n_jobs=4, time_bandwidth=2)
    tfr_epochs.apply_baseline(mode='mean', baseline=base_line)
    tfr_epochs.crop(crop_time[0], crop_time[1])

    return tfr_epochs, epochs

t_f = joblib.Parallel(n_jobs=15)(
    joblib.delayed(time_freq)(file) for file in file_includes)
    #joblib.delayed(time_freq)(epochs) for epochs in [i[3] for i in l_r_ep])
#%%
np.save(join(constants.BASE_DIRECTORY, 'cue_tf_data.npy'), t_f, allow_pickle=True)

#%%d
t_f = np.load(join(constants.BASE_DIRECTORY, 'cue_tf_data.npy'), allow_pickle=True)
t_f_e = [i[1] for i in t_f]
t_f_etf = [i[0] for i in t_f]

#%%

def TFR_concat(tfr_list):
    info = tfr_list[0].info
    times = tfr_list[0].times
    freqs = tfr_list[0].freqs
    comment = tfr_list[0].comment
    method = tfr_list[0].method
    events = tfr_list[0].events
    event_id = tfr_list[0].event_id
    selection = tfr_list[0].selection
    drop_log = tfr_list[0].drop_log
    metadata = tfr_list[0].metadata

    data = np.zeros([np.sum([len(i) for i in tfr_list])]+ list(tfr_list[0].data.shape[1:4]))
    e_ind = 0
    for i, ep in enumerate(tfr_list):
        data[e_ind:e_ind+len(ep), :,:,:] = ep.data
        e_ind += len(ep)

    comb_tfr = mne.time_frequency.EpochsTFR(data=data, info=info, times=times, freqs=freqs, comment=comment, method=method,
                                            events=events, event_id=event_id, selection=selection, drop_log=drop_log,
                                            metadata=metadata)
    return comb_tfr
comb = TFR_concat(t_f_etf)
comb.average().plot_joint()
plt.show()
comb2 = mne.concatenate_epochs(t_f_e)
comb2.crop(crop_time[0], crop_time[1])
comb2.average().plot_joint()
plt.show()

#%% channel and time window selector
space_inds = [39, 46, 47, 48, 49, 98]
time_inds =[590, 591, 592, 593, 594, 595]
og_sf = 250
ch_names = [nme for ind, nme in enumerate(t_f_etf[0].ch_names) if ind in space_inds]
for tfep in t_f_etf:
    tfep.pick_channels(ch_names)
    #tfep.crop(time_inds[0]/og_sf, time_inds[1]/og_sf)
#%% replicate analysis above in time frequency space
selectors = ['cue_dir == -1', 'cue_dir == 0 or cue_dir == 1']
# get data, format in participants x times x channels x freqs
X_N = np.zeros([len(t_f_etf)] + [t_f_etf[0].data.shape[3], t_f_etf[0].data.shape[1], t_f_etf[0].data.shape[2]])
X_C = np.zeros([len(t_f_etf)] + [t_f_etf[0].data.shape[3], t_f_etf[0].data.shape[1], t_f_etf[0].data.shape[2]])
for i in range(len(t_f_etf)):
    X_N[i, :, :] = np.transpose(t_f_etf[i][selectors[0]].average().data, (2,0,1))
    X_C[i, :, :] = np.transpose(t_f_etf[i][selectors[1]].average().data, (2,0,1))
#%% connectivity strutctrue different for tf
sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(
    t_f_etf[0].info, sen)

# we
adjacency = mne.stats.combine_adjacency(len(t_f_etf[0].times), sensor_adjacency, len(t_f_etf[0].freqs))
#adjacency = mne.stats.combine_adjacency(len(t_f_etf[0].times), len(t_f_etf[0].freqs))
#%% Carry out the cluster permutation
threshold = 14.9
n_permutations = 100  # Warning: 50 is way too small for real-world analysis.
#threshold = dict(start=4., step=0.1)
# set family-wise p-value
p_accept = 0.05
sigma = 1e-3
stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
data = np.mean((X_C - X_N), axis=2)
data = (X_C - X_N)
data = [X_C.transpose((0,1,3,2)), X_N.transpose((0,1,3,2))]
data = [X_C, X_N]
# cluster_stats = mne.stats.perm(data, n_permutations=n_permutations,
#                                              threshold=threshold, tail=0,
#                                              n_jobs=12, buffer_size=None)

cluster_stats = mne.stats.permutation_cluster_test(data,
                                                   threshold=threshold, n_permutations=n_permutations,
                                                   tail=0, adjacency=None, n_jobs=12)

T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]
print(good_cluster_inds)
print(p_values.min())
print([np.unique(clusters[i][0]) for i in good_cluster_inds])

#%% plot the time-freq chart and mask off
# Create new stats image with only significant clusters
av_map = np.mean((X_C - X_N), axis=2).mean(axis=0).transpose()
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, p_values):
    if p_val <= 0.1:
        T_obs_plot[c] = T_obs[c]

plt.imshow(np.nanmean(T_obs, axis=1).transpose(), cmap=plt.cm.gray)
plt.imshow(np.nanmean(T_obs_plot, axis=1).transpose())
plt.show()

# plt.imshow(T_obs, cmap=plt.cm.gray)
# plt.imshow(T_obs_plot)
# plt.show()

#%% using this cluster as a mask, does the summed value predict outcomes?

#for each participants mask off the difference
time_inds, space_inds = np.squeeze(clusters[2])
mask_array = np.zeros(data.shape[0], dtype=int)

ind_value = []
for i in range(data.shape[0]):


#%% Does this time-course predict the accuracy of participants?
# read in meta data and trial data
meta = pd.read_csv(join(constants.BASE_DIRECTORY, 'worms_meta.csv'))
meta['id'] = meta.Alex_ID
all_trials = pd.read_csv(join(constants.BASE_DIRECTORY, 'all_trials.csv'))
all_trials['id'] = all_trials.id.astype(str)
# left or right
glm_data = copy.copy(data)

#file_includes_even = file_includes[0:-1]
#glm_data = glm_data[0:-1]

#predictors to use (column names for the all_trials)
keys = ['Overall Error', 'Overall Precision', 'Overall Onset', 'Overall Duration']
src = [meta] * len(keys)
reg_data = np.zeros((len(glm_data), len(keys)))
for i, key in enumerate(keys):
    for ii, filename in enumerate(file_includes):
        _id = int(filename.split('_')[0])
        reg_data[ii,i] = src[i][src[i].id == _id][key].abs().mean()

        for iii in range(reg_data.shape[1]):
            reg_data[np.isnan(reg_data[:,iii]),iii] = np.nanmean(reg_data[:,iii])

# build glm
# regressors
regs = list()
regs.append(glmtools.regressors.ConstantRegressor(num_observations=len(glm_data)))
for i, name in enumerate(keys):
    regs.append(glmtools.regressors.ParametricRegressor(values=reg_data[:,i],
                                                        name=name,
                                                        preproc='z',
                                                        num_observations=len(glm_data)))
# contrasts
contrasts = list()
names = ['Intercept'] + keys
for i, _reg in enumerate(regs):
    cont_vals = [0] * (len(keys)+1)
    cont_vals[i] = 1
    contrasts.append(glmtools.design.Contrast(name=names[i], values=cont_vals))


#data
dat = glmtools.data.TrialGLMData(data=glm_data, dim_labels=['participants', 'times', 'sensors'])

#design matrix
des = glmtools.design.GLMDesign.initialise(regs,contrasts)

# model
model = glmtools.fit.OLSModel( des, dat )

#%% plot betas
i = 1
dum_ev = mne.EvokedArray(np.transpose(model.betas[i]), info=evs[0][0].info, tmin=evs[0][0].times[0])
mne.viz.plot_compare_evokeds(dum_ev,truncate_yaxis='auto', combine='mean')
#%% cluster permutation on MEG data
# Cluster permutations - detect contiguous regions containing effects.
perm_args = {'cluster_forming_threshold': 0.5, 'pooled_dims': (1, 2)}
perms = 100
i = 2
CP = glmtools.permutations.ClusterPermutation(des, dat, i, perms, perm_args=perm_args, metric='tstats', nprocesses=5)
cluster_masks, cluster_stats = CP.get_sig_clusters(dat, 95)
print(cluster_stats)

#%%

#%% permute the beta weights from the OLS regression
perms = 100
sig_thresh = 95
sig_mask = []
CPs =[]
for i in range(1,len(regs)):
    CP = glmtools.permutations.Permutation(des, dat, i, perms, metric='betas', nprocesses=12)
    results = CP.get_sig_at_percentile(sig_thresh)
    sig_mask.append(results)
    CPs.append(CP)
sig_mask = np.array(sig_mask)

#%% get the cluster

clus_ind = 0

cluster_mask = (cluster_masks == clus_ind).astype(int)

plt_data = model.tstats[i]

time_inds = np.where(cluster_mask.mean(axis=1) > 0)[0]
ch_inds = np.where(cluster_mask.mean(axis=0) > 0)[0]
# get topography for F stat

f_map = plt_data[time_inds, ...].mean(axis=0)

# get signals at the sensors contributing to the cluster
sig_times = L_E[0].times[time_inds]
# get sensor positions via layout
pos = mne.find_layout(L_E[0].info).pos

picks, pos, merge_channels, names, ch_type, sphere, clip_origin = \
    mne.viz.topomap._prepare_topomap_plot(L_E[0].info, sen)


# create spatial mask
mask = np.zeros((f_map.shape[0], 1), dtype=bool)
mask[ch_inds, :] = True
#reg_masks.append(mask)
# initialize figure
fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

# plot average test statistic and mark significant sensors
image, _ = mne.viz.plot_topomap(f_map, pos, mask=mask, axes=ax_topo, ch_type=sen,
                                vmin=np.min, vmax=np.max, sphere=sphere, show=False)

# create additional axes (for ERF and colorbar)
divider = make_axes_locatable(ax_topo)

# add axes for colorbar
ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(image, cax=ax_colorbar)
ax_topo.set_xlabel(
    'Averaged T-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

# add new axis for time courses and plot time courses
ax_signals = divider.append_axes('right', size='300%', pad=1.2)
title = f'Retro-Cue ~ {model.contrast_names[i]} Cluster #{i_clu + 1}, {len(ch_inds)} sensor'
if len(ch_inds) > 1:
    title += "s (mean)"

dum_ev = mne.EvokedArray(np.transpose(plt_data), info=L_E[0].info, tmin=L_E[0].times[0])
mne.viz.plot_compare_evokeds(dum_ev, title=title, picks=ch_inds, axes=ax_signals, show=False,
                             split_legend=True, truncate_yaxis='auto', combine='mean')

ax_signals.set_title(title, y=1.1)
# plot temporal cluster extent
ymin, ymax = ax_signals.get_ylim()
ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                         color='orange', alpha=0.3)

xmin, xmax = ax_signals.get_xlim()

# add labels if needed
for i, ev in enumerate(event_labels):
    if xmin <= event_onsets[i] <= xmax:
        ax_signals.axvline(event_onsets[i], linestyle='--')
        ax_signals.text(event_onsets[i], ymax, f'{ev} onset')
    if xmin <= event_offsets[i] <= xmax:
        ax_signals.axvline(event_offsets[i], linestyle='--')
        ax_signals.text(event_offsets[i], ymax, f'{ev} offset')

ax_signals.set_ylabel('Field Strength (fT)')
# clean up viz
mne.viz.tight_layout(fig=fig)
fig.subplots_adjust(bottom=.05)

labels = [item.get_text() for item in ax_signals.get_xticklabels()]
labels = [str(float(i) - plot_t_offset) for i in labels]
ax_signals.set_xticklabels(labels)
plt.show()


#%% load epochs for each participant and the crop to postcue period flip left to right

reg_n = 'ang_dist'
# 0 = stim, 1 = maintenance, 2 = cue, 2.5 = postcue, 3.5 = probe,
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

def decode_probe(file):
    #load in epochs
    _id = file.split('_')[0]
    try:
        epochs = mne.epochs.read_epochs(join(epodir, file))
        if len(epochs) < 30:
            return [0]
        # crop out the postcue period after pre-stim baseline
        # epochs.apply_baseline(baseline=(3, 3.5))
        # epochs.crop(tmin=3, tmax=4.5)
        epochs.apply_baseline(baseline=(1.5, 2))
        epochs.crop(tmin=1.5, tmax=3.5)
        epochs.metadata = epochs.metadata.assign(Intercept=1)
        epochs.metadata['ang_dist'] = ang_dist(epochs.metadata[['targ_ang', 'resp_ang']], 90)
        epochs.pick_types(meg=True, chpi=False)
        #epochs.metadata.perc_diff = (epochs.metadata.perc_diff - epochs.metadata.perc_diff.mean) /epochs.metadata.perc_diff
        # seperate out probe directions

        _l = epochs['cue_dir == 0']
        _r = epochs['cue_dir == 1']
        _n = epochs['cue_dir == -1']
        lav = _l.average()
        rav = _r.average()
        nav = _n.average()

        # l = mne.combine_evoked([lav, nav], [1, -1])
        # r = mne.combine_evoked([rav, nav], [1, -1])

        l = _l
        r = _r
        names = ["Intercept", reg_n]
        res_l = linear_regression(l, l.metadata[names].reset_index(drop=True), names=names)
        res_r = linear_regression(r, r.metadata[names].reset_index(drop=True), names=names)
        res_both = linear_regression(epochs, epochs.metadata[names].reset_index(drop=True), names=names)
        return [res_l, res_r, res_both]
    except:
        return [0]

id_scores_coeff = joblib.Parallel(n_jobs=15)(
    joblib.delayed(decode_probe)(file) for file in file_includes)

#%% calculate average coeeficiants
l_int = []; r_int = []; l_acc = []; r_acc = []
e_int = []; e_acc = []
for res in id_scores_coeff:
    if res != [0]:
        l_int.append(res[0]['Intercept'].beta)
        r_int.append(res[1]['Intercept'].beta)
        e_int.append(res[2]['Intercept'].beta)
        l_acc.append(res[0][reg_n].beta)
        r_acc.append(res[1][reg_n].beta)
        e_acc.append(res[2][reg_n].beta)

#%%
ldata = np.array([i.data for i in l_acc])
rdata = np.array([i.data for i in r_acc])
edata = np.array([i.data for i in e_acc])

plot_l = mne.EvokedArray(ldata.mean(axis=0), l_int[0].info, tmin=-0.5,
                               nave=ldata.shape[0], comment='l average')
plot_r = mne.EvokedArray(rdata.mean(axis=0), r_int[0].info, tmin=-0.5,
                               nave=rdata.shape[0], comment='r average')

plot_e = mne.EvokedArray(edata.mean(axis=0), r_int[0].info, tmin=-0.5,
                               nave=rdata.shape[0], comment='Epoch average')

plot_e.plot_joint()
plot_r.plot_joint()
plot_l.plot_joint()

#mne.viz.plot_compare_evokeds([plot_l, plot_r])

#%% Cluster t test on regression betas
ch_type='mag'
epoch_data = copy.deepcopy(e_acc)
epoch_data = [i.pick_types(meg=ch_type) for i in epoch_data]

adjacency = mne.channels.find_ch_adjacency(r_int[0].info, ch_type=ch_type)

# select channels
#np.transpose(l_r[i][0].data, (1,0))
edata = np.array([np.transpose(i.data, (1,0)) for i in epoch_data])

threshold = 3.5  # very high, but the test is quite sensitive on this data
#threshold = dict(start=0, step=0.2)
# set family-wise p-value
p_accept = 0.05
sigma = 1e-3
stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test(edata, n_permutations=1000,
                                             threshold=threshold, tail=0,
                                             n_jobs=10, buffer_size=None,
                                             adjacency=adjacency[0], stat_fun=stat_fun_hat)

T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]
print(good_cluster_inds)
print(p_values.min())