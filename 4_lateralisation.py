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

epodir = join(constants.BASE_DIRECTORY, 'new_epochs_2')
files = [i for i in listdir(epodir) if 'metastim' in i]
ids = [i.split('_')[0] for i in files]

#%% Select files with 'good' visual evoked responses
file_includes = np.load(join(constants.BASE_DIRECTORY, 'good_visual_evoked.npy'))
#file_includes = files
epochs = mne.epochs.read_epochs(join(epodir, file_includes[0]))
epochs.apply_baseline(baseline=(2,2.5))
epochs.crop(tmin=1.5, tmax=3.5)
freqs = np.arange(4., 35., 1.)
n_cycles = freqs / 2.
power = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs,
                   n_cycles=n_cycles, return_itc=False, average=False)

#%%
# power.plot_topo(mode='logratio', title='Average power')
av_power_l = power['cue_dir == 1'].average()
av_power_r = power['cue_dir == 0'].average()
av_power_diff = mne.combine_evoked([av_power_l, av_power_r], [1,-1])
av_power_diff.plot_topomap(ch_type='mag', tmin=2.5, tmax=3.5, fmin=0, fmax=100, mode='logratio',
                   title='Alpha', show=True)
#%% just make an average evoked period
"""
TIMINGS 
0 - STIMULUS: duration 1
1 - MAINTENANCE: duration 1
2 - CUE: duration 0.5
2.5 - POSTCUE: duration 1
3.5 - PROBE
"""
sen = 'mag'

event_labels = ['Stimulus', 'Cue', 'Probe']
event_onsets = [0, 2, 3.5]
event_offsets = [1, 2.5, 4.496]
bad_thresh = 10
def av_evoked(file):
    try:
        epochs = mne.epochs.read_epochs(join(epodir, file))
        epochs.pick_types(meg=sen)
        epochs.apply_baseline(baseline=(1.5, 2))
        epochs.crop(tmin=1.5, tmax=3.5)

        #detect artefacts in epochs and reject if at more than one timepoint
        bad_es = []
        for i in range(len(epochs)):
            bad_es.append(sails.utils.detect_artefacts(epochs[i]._data[0], axis=1,
                                     reject_mode='dim',
                                     ret_mode='bad_inds').sum())
        bad_es_mask = np.array(bad_es) > bad_thresh
        epochs = epochs[~bad_es_mask]
        epochs = epochs.equalize_event_counts(['cue_dir == 0', 'cue_dir == 1', 'cue_dir == -1'])[0]
        l = epochs['cue_dir == 0']
        r = epochs['cue_dir == 1']
        n = epochs['cue_dir == -1']
        cued = epochs['cue_dir == 0 or cue_dir == 1']
        lav = l.average()
        rav = r.average()
        nav = n.average()
        cav = cued.average()
        l_cue = mne.combine_evoked([lav, nav], [1,-1])
        r_cue = mne.combine_evoked([rav, nav], [1, -1])
        cue = mne.combine_evoked([cav, nav], [1, -1])
    except:
        print(file)
        print(epochs)
    #return(lav,rav)
    return(lav, rav, nav)

    #return(l_cue, r_cue)

l_r = joblib.Parallel(n_jobs=15)(
    joblib.delayed(av_evoked)(file) for file in file_includes)

# l_r = []
# for f in file_includes:
#     l_r.append(av_evoked(f))
print([i[2] for i in l_r])
#%%
for pair in l_r:
    pair[0].plot_joint()

#%% cluster test
X_L = np.empty((len(l_r), len(l_r[0][0].times), l_r[0][0].data.shape[0]))
X_R = np.empty((len(l_r), len(l_r[0][0].times), l_r[0][0].data.shape[0]))
X_N = np.empty((len(l_r), len(l_r[0][0].times), l_r[0][0].data.shape[0]))
L_E = []
R_E = []
N_E = []
for i in range(len(l_r)):
    X_L[i,:,:] = np.transpose(l_r[i][0].data, (1,0))
    X_R[i,:,:] = np.transpose(l_r[i][1].data, (1,0))
    X_N[i, :, :] = np.transpose(l_r[i][2].data, (1, 0))
    L_E.append(l_r[i][0])
    R_E.append(l_r[i][1])
    N_E.append(l_r[i][2])

#connectivity strutctrue
adjacency = mne.channels.find_ch_adjacency(l_r[0][0].info, ch_type=sen)

threshold = 3  # very high, but the test is quite sensitive on this data
#threshold = dict(start=4, step=0.5)
# set family-wise p-value
p_accept = 0.05
sigma = 1e-3
stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test((X_L - X_N), n_permutations=1000,
                                             threshold=threshold, tail=0,
                                             n_jobs=10, buffer_size=None,
                                             adjacency=adjacency[0], stat_fun=stat_fun_hat)

T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]
print(good_cluster_inds)
print(p_values.min())
print([np.unique(clusters[i][0]) for i in good_cluster_inds])
#%% Plot clusters
labels_plot = ['Left Cue', 'Neutral Cue']
evs = [L_E, N_E]
colors = {labels_plot[0]: "crimson", labels_plot[1]: 'steelblue'}
linestyles = {"L": '-', "R": '--'}

# get sensor positions via layout
pos = mne.find_layout(L_E[0].info).pos

picks, pos2, merge_channels, names, ch_type, sphere, clip_origin = \
    mne.viz.topomap._prepare_topomap_plot(L_E[0].info, sen)

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
    sig_times = L_E[0].times[time_inds]

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True
    reg_masks.append(mask)
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
    title = 'Retro-Cue Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"
    mne.viz.plot_compare_evokeds({labels_plot[0]:evs[0], labels_plot[1]:evs[1]}, title=title, picks=ch_inds, axes=ax_signals,
                         colors=colors, show=False,
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
            ax_signals.axvline(event_onsets[i], linestyle = '--')
            ax_signals.text(event_onsets[i], ymax, f'{ev} onset')
        if xmin <= event_offsets[i] <= xmax:
            ax_signals.axvline(event_offsets[i], linestyle = '--')
            ax_signals.text(event_offsets[i], ymax, f'{ev} offset')

    ax_signals.set_ylabel('Field Strength (fT)')
    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)

    labels = [item.get_text() for item in ax_signals.get_xticklabels()]
    labels = [str(float(i)-plot_t_offset) for i in labels]
    #ax_signals.set_xticklabels(labels)
    plt.show()


#%% Does this time-course predict the accuracy of participants?
# read in meta data and trial data
meta = pd.read_csv(join(constants.BASE_DIRECTORY, 'Combined3.csv'))
all_trials = pd.read_csv(join(constants.BASE_DIRECTORY, 'all_trials.csv'))

# left or right
glm_data = copy.copy(X_L)

file_includes_even = file_includes[0:-1]
glm_data = glm_data[0:-1]

#predictors to use (column names for the all_trials)
keys = ['ang_dist']
reg_data = np.zeros((len(glm_data), len(keys)))
for i, key in enumerate(keys):
    for ii, filename in enumerate(file_includes_even):
        _id = int(filename.split('_')[0])
        reg_data[ii,i] = all_trials[all_trials.id == _id][key].mean()
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

#%% cluster permutation on MEG data
# Cluster permutations - detect contiguous regions containing effects.
perm_args = {'cluster_forming_threshold': 1, 'pooled_dims': (1, 2)}
perms = 100
i = 0
CP = glmtools.permutations.ClusterPermutation(des, dat, i, perms, perm_args=perm_args, metric='tstats', nprocesses=5)
cluster_masks, cluster_stats = CP.get_sig_clusters(dat, 95)
print(cluster_stats)

#%% get the cluster

clus_ind = 1

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