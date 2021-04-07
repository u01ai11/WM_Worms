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

epodir = join(constants.BASE_DIRECTORY, 'new_epochs')
files = [i for i in listdir(epodir) if 'metastim' in i]
ids = [i.split('_')[0] for i in files]

#%% Select files with 'good' visual evoked responses
file_includes = np.load(join(constants.BASE_DIRECTORY, 'good_visual_evoked.npy'))

#%% just make an average evoked period
"""
TIMINGS 
0 - STIMULUS: duration 1
1 - MAINTENANCE: duration 1
2 - CUE: duration 0.5
2.5 - POSTCUE: duration 1
3.5 - PROBE
"""
sen = 'grad'
def av_evoked(file):
    epochs = mne.epochs.read_epochs(join(epodir, file))
    epochs.pick_types(meg=sen)
    epochs.apply_baseline(baseline=(-0.5, 0))
    #epochs.crop(tmin=1.5, tmax=3.5)
    l = epochs['cue_dir == 0']
    r = epochs['cue_dir == 1']
    return(l.average(), r.average())

l_r = joblib.Parallel(n_jobs=15)(
    joblib.delayed(av_evoked)(file) for file in file_includes)

#%% cluster test
X_L = np.empty((len(l_r), len(l_r[0][0].times), l_r[0][0].data.shape[0]))
X_R = np.empty((len(l_r), len(l_r[0][0].times), l_r[0][0].data.shape[0]))
L_E = []
R_E = []
for i in range(len(l_r)):
    X_L[i,:,:] = np.transpose(l_r[i][0].data, (1,0))
    X_R[i,:,:] = np.transpose(l_r[i][1].data, (1,0))
    L_E.append(l_r[i][0])
    R_E.append(l_r[i][1])

#connectivity strutctrue
adjacency = mne.channels.find_ch_adjacency(l_r[0][0].info, ch_type=sen)

threshold = 3  # very high, but the test is quite sensitive on this data
#threshold = dict(start=0, step=0.2)
# set family-wise p-value
p_accept = 0.05
sigma = 1e-3
stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test((X_L-X_R), n_permutations=1000,
                                             threshold=threshold, tail=0,
                                             n_jobs=1, buffer_size=None,
                                             adjacency=adjacency[0], stat_fun=stat_fun_hat)

T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]
print(good_cluster_inds)
print(p_values.min())
#%% Plot clusters
colors = {"Left": "crimson", "Right": 'steelblue'}
linestyles = {"L": '-', "R": '--'}

# get sensor positions via layout
pos = mne.find_layout(L_E[0].info).pos

picks, pos2, merge_channels, names, ch_type, sphere, clip_origin = \
    mne.viz.topomap._prepare_topomap_plot(L_E[0].info, sen)

reg_masks = []
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
    title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"
    mne.viz.plot_compare_evokeds({'Left':L_E, 'Right':R_E}, title=title, picks=ch_inds, axes=ax_signals,
                         colors=colors, show=False,
                         split_legend=True, truncate_yaxis='auto', combine='mean')

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)

    ylims = ax_signals.get_ylim()

    ax_signals.set_ylabel('Field Strength (fT)')
    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()


#%% plot them




#%% load epochs for each participant and the crop to postcue period flip left to right

# 0 = stim, 1 = maintenance, 2 = cue, 2.5 = postcue, 3.5 = probe,

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
        epochs.apply_baseline(baseline=(-0.5, 0))
        epochs.metadata = epochs.metadata.assign(Intercept=1)
        epochs.pick_types(meg=True, chpi=False)

        # seperate out probe directions
        l = epochs['targ == 0']
        r = epochs['targ == 1']

        names = ["Intercept", 'perc_diff']
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
        l_acc.append(res[0]['perc_diff'].beta)
        r_acc.append(res[1]['perc_diff'].beta)
        e_acc.append(res[2]['perc_diff'].beta)

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

threshold = 3  # very high, but the test is quite sensitive on this data
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