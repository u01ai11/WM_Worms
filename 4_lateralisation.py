"""
5_lateralisation.py
Alex Anwyl-Irvine 2021

Over the group:
    1. Check if we can decode from the probe, discard if not
    2. If passed look at post cue alpha

"""

from os.path import join
from os import listdir
import numpy as np
import mne
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import (SlidingEstimator, cross_val_multiscore)
import joblib
import pandas as pd
import matplotlib.pyplot as plt
try:
    import constants
    from REDTools import epoch
except:
    import sys
    sys.path.insert(0, '/home/ai05/WM_Worms')
    import constants


#%% load in participant epochs to check probe
epodir = join(constants.BASE_DIRECTORY, 'epoched')
probe_files =[f for f in listdir(epodir) if 'probe' in f]
probe_ids = list(set([f.split('_')[0] for f in probe_files]))
probe_ids.sort()

good_ids = []
def decode_probe(_id):
    #load in epochs
    print(_id)
    try:
        file = [f for f in probe_files if _id in f][0]
        epochs = mne.epochs.read_epochs(join(epodir, file))
        if len(epochs) < 20:
            return [0,0]
        epochs.crop(tmin=0,tmax=1)
        epochs.filter(2,20)
        epochs.resample(200)
        X = epochs.get_data()  # MEG signals: n_epochs, n_meg_channels, n_times
        y = epochs.events[:, 2]  # target: left vs right probes
        clf = make_pipeline(StandardScaler(), mne.decoding.LinearModel(LogisticRegression(solver='lbfgs')))
        time_decod = SlidingEstimator(clf, n_jobs=5, scoring='roc_auc', verbose=True)
        scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=1)
        scores = np.mean(scores, axis=0)
        time_decod.fit(X,y)
        coef = mne.decoding.get_coef(time_decod, 'patterns_', inverse_transform=True)
        good_ids.append(_id)
        return [scores, coef]
    except:
        return [0,0]

id_scores_coeff = joblib.Parallel(n_jobs=15)(
    joblib.delayed(decode_probe)(_id) for _id in probe_ids)

#%% illustrate decoding accuracy
id_scores_coeff_c = [i for i in id_scores_coeff if len(i) >1]
id_scores_coeff_c = [i for i in id_scores_coeff_c if len(str(i[0])) > 1]
id_scores = [i[0] for i in id_scores_coeff_c]
coeff_scores = [i[1] for i in id_scores_coeff_c]
#%%
plt_scores = np.array(id_scores)
fig, ax = plt.subplots()
time = np.linspace(0,1,num=plt_scores.shape[1])
for line in plt_scores:
        ax.plot(time,line)

ax.plot(time,plt_scores.mean(axis=0), color='k', label='mean')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Left vs Right Probe decoding')
plt.tight_layout()
plt.show()

#%% illustrate the coefficients
mean_coeff = np.array(coeff_scores)
mean_coeff = mean_coeff.mean(axis=0)

e = mne.epochs.read_epochs(join(epodir, probe_files[0]))
e.crop(tmin=0, tmax=1); e.filter(2, 20); e.resample(200)
evoked_time_gen = mne.EvokedArray(mean_coeff, e.info, tmin=e.times[0])
evoked_time_gen.plot_joint()
#%% save good ids
good_ids = np.array(probe_ids)[np.array([len(i)>1 for i in id_scores])]
np.save(join(constants.BASE_DIRECTORY, f'decoded_ids.npy'),good_ids)

#%% proceed with alpha lateralisation analysis using these participants!
cue_files = [f for f in listdir(epodir) if 'postcue' in f]
good_ids = np.load(join(constants.BASE_DIRECTORY, f'decoded_ids.npy'))

def get_evoked(_id, name):
    try:
        file = [i for i in cue_files if _id in i][0]
        epochs = mne.epochs.read_epochs(join(epodir, file))
        evoked = epochs[name].average()
        return evoked
    except:
        return None

left = joblib.Parallel(n_jobs=5)(
    joblib.delayed(get_evoked)(_id, 'L_CUE') for _id in good_ids)
left = [i for i in left if i!=None]
right = joblib.Parallel(n_jobs=5)(
    joblib.delayed(get_evoked)(_id, 'R_CUE') for _id in good_ids)
right = [i for i in right if i!=None]
neutral  = joblib.Parallel(n_jobs=5)(
    joblib.delayed(get_evoked)(_id, 'N_CUE') for _id in good_ids)
neutral = [i for i in neutral if i!=None]
#%% have a look
sel = mne.read_selection(name='Left-occipital', info=left[1].info)
mne.viz.plot_compare_evokeds({'Left':left, 'Right':right, 'Neutral':neutral},
                             picks=sel,
                             combine='mean',
                             show_sensors=True,
                             title='Post cue - Left')

sel = mne.read_selection(name='Right-occipital', info=left[1].info)
mne.viz.plot_compare_evokeds({'Left':left, 'Right':right, 'Neutral':neutral},
                             picks=sel,
                                    combine='mean',
                                    show_sensors=True,
                             title='Post cue - Right')

#%% Sort out per participant data

#%% perform cluster test to identify sensors
# load in epochs
cue_files = [f for f in listdir(epodir) if 'postcue' in f]
cue_files = [f for f in cue_files if f.split('_')[0] in good_ids]
all_eps = []
for f in cue_files:
    try:
        all_eps.append(mne.read_epochs(join(epodir, f)))
    except:
        print(f)

sub_e = []
#subtract pairwise and match
for eps in all_eps:
    # calcullate evokes
    eps.filter(2, 50)
    eps.resample(200)
    le_e = eps['L_CUE'].average()
    ri_e = eps['R_CUE'].average()
    sub_e.append(mne.combine_evoked([le_e,ri_e], weights=[1,-1]))


decim = 6
freqs = np.arange(8, 40, 4)  # define frequencies of interest
sfreq = l_e.info['sfreq']  # sampling in Hz
tfr_l = mne.time_frequency.tfr_morlet(l_e, freqs, n_cycles=4., decim=decim,
                        average=False, return_itc=False, n_jobs=10)
tfr_r = mne.time_frequency.tfr_morlet(r_e, freqs, n_cycles=4., decim=decim,
                        average=False, return_itc=False, n_jobs=10)

# Baseline power
tfr_l.apply_baseline(mode='logratio', baseline=(-0.5, 0.0))
# Baseline power
tfr_r.apply_baseline(mode='logratio', baseline=(-0.5, 0.0))
#save
tfr_l.save(join(constants.BASE_DIRECTORY, 'epoched','tfr_l-epo.fif'), overwrite=True)
tfr_r.save(join(constants.BASE_DIRECTORY, 'epoched','tfr_r-epo.fif'), overwrite=True)
#%% do mags ony first
chans = 'grad'
tfr_r.pick_types(meg=chans)
tfr_l.pick_types(meg=chans)
#%%
sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(
    tfr_r.info,chans)
adjacency = mne.stats.combine_adjacency(
    sensor_adjacency, len(tfr_r.freqs), len(tfr_r.times))
threshold = 3.
n_permutations = 50  # Warning: 50 is way too small for real-world analysis.

epochs_power = tfr_l.data[:tfr_r.data.shape[0],:,:,:] - tfr_r.data
T_obs, clusters, cluster_p_values, H0 = \
    mne.stats.permutation_cluster_1samp_test(epochs_power, n_permutations=n_permutations,
                                   threshold=threshold, tail=0,
                                   adjacency=adjacency,
                                   out_type='mask', verbose=True)
#%% Try without power
sub_e = []
#subtract pairwise and match
for eps in all_eps:
    eps.filter(2, 50, n_jobs=15)
    eps.resample(200, n_jobs=15)

    #eps.apply_baseline(baseline=(0.5,0.6))
    le_e = eps['L_CUE'].average()
    ri_e = eps['R_CUE'].average()
    sub_e.append(mne.combine_evoked([le_e,ri_e], weights=[1,-1]))
av = mne.combine_evoked(sub_e, weights='nave')

#%% Try looking at TFR
np.save(join(constants.BASE_DIRECTORY, 'evoked', 'postcue-evo.npy'), all_eps, allow_pickle=True)
all_eps = np.load(join(constants.BASE_DIRECTORY, 'evoked', 'postcue-evo.npy'))
#%%
decim = 3
freqs = np.arange(7, 40, 2)  # define frequencies of interest
sfreq = eps.info['sfreq']  # sampling in Hz
sub_e_t = []
for eps in all_eps:
    tfr_eps = mne.time_frequency.tfr_morlet(eps, freqs, n_cycles=4., decim=decim,
                                          average=False, return_itc=False, n_jobs=10)
    tfr_l = tfr_eps['L_CUE'].average()
    tfr_r = tfr_eps['R_CUE'].average()
    sub_e_t.append(mne.combine_evoked([tfr_l, tfr_r], weights=[1, -1]))

av_t = mne.combine_evoked(sub_e_t, weights='nave')
#%%
av_t.plot_joint(baseline=(0.5,0.6), picks='mag', mode='mean', timefreqs=[(0.7,8)])
#%% investiage only kids with good lateralisation
for i, e in enumerate(sub_e):
    e.plot_joint(picks='mag', title=str(i))
#%%
goods = [0,1,3,5,9,11,15,16,18,19,20,21,22,23,24,25,26,27,28,29,35,36,37,38,39]
av_g = mne.combine_evoked([sub_e[i] for i in goods], weights='nave')
av_g.plot_joint()
#%%
chans = 'mag'
sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(
    av.info,chans)
adjacency = mne.stats.combine_adjacency(
    sensor_adjacency, len(av.times))
threshold = 3
n_permutations = 100  # Warning: 50 is way too small for real-world analysis.
av.pick_types(meg=chans)
X = np.zeros([len(sub_e),av.data.shape[0], av.data.shape[1]])
for i, d in enumerate(sub_e):
    _d = d.copy()
    _d.pick_types(meg=chans)
    X[i,:,:] = _d.data

epochs_power = X
T_obs, clusters, cluster_p_values, H0 = \
    mne.stats.permutation_cluster_1samp_test(epochs_power, n_permutations=n_permutations,
                                   threshold=threshold, tail=0,
                                   adjacency=adjacency,
                                   out_type='mask', verbose=True)


#%%TFR
sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(
    sub_e_t[0].info,chans)
adjacency = mne.stats.combine_adjacency(
    sensor_adjacency, len(sub_e_t[0].freqs), len(sub_e_t[0].times))
threshold = 3.
n_permutations = 50  # Warning: 50 is way too small for real-world analysis.

X = np.zeros([len(sub_e_t),102, 17,67])
for i, d in enumerate(sub_e_t):
    _d = d.copy()
    _d.pick_types(meg='mag')
    X[i,:,:,:] = _d.data

epochs_power =X
T_obs, clusters, cluster_p_values, H0 = \
    mne.stats.permutation_cluster_1samp_test(epochs_power, n_permutations=n_permutations,
                                   threshold=threshold, tail=0,
                                   adjacency=adjacency,
                                   out_type='mask', verbose=True)

#%% load in participant epochs to decode
epodir = join(constants.BASE_DIRECTORY, 'epoched')
probe_files =[f for f in listdir(epodir) if 'wholecue' in f]
probe_ids = list(set([f.split('_')[0] for f in probe_files]))
probe_ids.sort()

def decode_probe(_id):
    #load in epochs
    print(_id)
    try:
        file = [f for f in probe_files if _id in f][0]
        epochs = mne.epochs.read_epochs(join(epodir, file))
        if len(epochs) < 20:
            return [0]
        epochs.filter(2,20)
        epochs.resample(200)
        epochs.apply_baseline()
        epochs.pick_types(meg='mag')
        X = epochs['L_CUE', 'R_CUE'].get_data()  # MEG signals: n_epochs, n_meg_channels, n_times
        y = epochs['L_CUE', 'R_CUE'].events[:, 2]  # target: left vs right probes
        clf = make_pipeline(StandardScaler(), mne.decoding.LinearModel(LogisticRegression(solver='lbfgs')))
        time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc', verbose=True)
        time_decod.fit(X,y)
        coef = mne.decoding.get_coef(time_decod, 'patterns_', inverse_transform=True)

        scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=1)
        scores = np.mean(scores, axis=0)
        evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
        return (scores, coef)
    except:
        return [0]

scores_coef = joblib.Parallel(n_jobs=15)(
    joblib.delayed(decode_probe)(_id) for _id in probe_ids)
scores_coef = [i for i in scores_coef if len(i) >1]
id_scores = [i[0] for i in scores_coef]
id_coef = [i[1] for i in scores_coef]
#%% illustrate decoding
plt_scores = np.array([i for i in id_scores if len(i) > 1])
fig, ax = plt.subplots()
time = np.linspace(0,1.5,num=plt_scores.shape[1])
for line in plt_scores:
    ax.plot(time,line, linewidth=0.5)

ax.plot(time,plt_scores.mean(axis=0), color='k', label='mean')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Left vs Right Probe decoding during maintenance')
plt.tight_layout()
plt.show()


#%%
"""
rescore probed left vs right 



"""

#%%
id_array = np.array(id_coef)
file = probe_files[1]
epochs = mne.epochs.read_epochs(join(epodir, file))

epochs.filter(2, 20)
epochs.resample(200)
epochs.pick_types(meg='mag')

evoked_time_gen = mne.EvokedArray(id_array.mean(axis=0), epochs.info, tmin=0)
evoked_time_gen.plot_joint()

#%% check files and IDs

epodir = join(constants.BASE_DIRECTORY, 'epoched')
probe_files =[f for f in listdir(epodir) if 'postcue' in f]
probe_ids = list(set([f.split('_')[0] for f in probe_files]))
probe_ids.sort()

def look_epochs(_id):
    file = [f for f in probe_files if _id in f][0]
    epochs = mne.epochs.read_epochs(join(epodir, file))
    return epochs


epos = joblib.Parallel(n_jobs=15)(
    joblib.delayed(look_epochs)(_id) for _id in probe_ids)

#%%
epo_len = [len(i) for i in epos]
datadir = join(constants.BASE_DIRECTORY, 'behav')
data_f = listdir(datadir)
part_f = [[ii for ii in data_f if i in ii]for i in probe_ids]
part_f = [i if len(i) > 1 else ['','',''] for i in part_f]
[i.sort() for i in part_f]
part_f = [[l] +f for l,f in zip(epo_len, part_f)]
part_f = [[_id] + i for i, _id in zip(part_f, probe_ids)]
df = pd.DataFrame(part_f, columns=['ID', 'Epochs', 'Detailed', 'Events', 'Trials'])