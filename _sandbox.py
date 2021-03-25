"""
_sandbox.py
Alex Anwyl-Irvine 2021

basically a temporary file for checking bits of scripts work
"""

from os.path import join
from os import listdir
import numpy as np
import mne
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

#%% load in participant epochs to check probe
epodir = join(constants.BASE_DIRECTORY, 'epoched')
probe_files =[f for f in listdir(epodir) if 'probe' in f]
probe_ids = list(set([f.split('_')[0] for f in probe_files]))
probe_ids.sort()
datadir = join(constants.BASE_DIRECTORY, 'behav')

#%% load behavioural data
meta = pd.read_csv(join(constants.BASE_DIRECTORY, 'Combined3.csv'))
data_files = listdir(datadir)
data_trials = [i for i in data_files if 'trials' in i]
df = pd.read_csv(join(datadir, data_trials[1]), delimiter="\t")
all_trials = pd.DataFrame(columns=list(df.columns) + ['id'])
good_ids = []
behav_starts = pd.read_csv(join(datadir, 'dates.txt'), delimiter='\t', header=None, names=['name', 'time'])

for _id in probe_ids:
    _file = [f for f in data_trials if _id in f]
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

#
all_trials['offset'] = all_trials['targ_ang'] - all_trials['resp_ang']
all_trials['offset'] = all_trials['offset'].astype('float')
all_trials['offset_abs'] = all_trials['offset'].abs()
all_trials['cue_type'] = ['neutral' if i == 0 else 'valid' for i in all_trials['cue_dir']]
rolling_tmp = [all_trials[all_trials['id'] == i]['offset_abs'].rolling(25).mean() for i in all_trials['id'].unique()]
rolling_mean = []
for i in rolling_tmp:
    rolling_mean = rolling_mean + list(i)
all_trials['rolling_offset'] = rolling_mean
#%% Compare accuracy for cued vs non-cued trials

# Within subjects ANOVA
dv = 'resp_onset'#'perc_diff'
grp = 'cue_dir'
per_part = all_trials.groupby(['id', grp]).mean()
per_part = per_part.reset_index()
model = anova.AnovaRM(all_trials, depvar=dv,
                      subject='id', within=[grp], aggregate_func='mean')
res = model.fit()
print(res)

#posthocs
left = np.array(per_part[per_part['cue_dir'] == -1][dv])
right = np.array(per_part[per_part['cue_dir'] == -1][dv])
neutral = np.array(per_part[per_part['cue_dir'] == -1][dv])

l_r = weightstats.ttost_paired(left,right)


plt.close('all')
ax = sns.barplot(x=grp, y=dv, data=per_part,
                 linewidth=2.5, facecolor=(1, 1, 1, 0),
                 errcolor=".2", edgecolor=".2", ci='sd', capsize=0.2)
sns.stripplot(x=grp, y=dv, data=per_part, zorder=1,
              size=2, jitter=0.4,ax=ax)

plt.show()

#%% over time
plt.close('all')
#sns.lineplot(y='rolling_offset', x='trialnr', hue='id', data=all_trials, legend=False)
sns.lineplot(y='rolling_offset', x='trialnr', hue='cue_dir',data=all_trials)
plt.show()

#%% does average peformance predict WM assessments?
av_part = all_trials.groupby(['id']).mean()
av_part = av_part.reset_index()
c_name = 'WASI_Mat'
m_name = 'offset_abs'
covar = []
for i in av_part.id.unique():
    try:
        covar.append(float(meta[meta['Alex_ID'] == i][c_name]))
    except:
        covar.append(np.nan)
av_part[c_name] = covar
av_part[c_name] = (av_part[c_name] - av_part[c_name].mean())/av_part[c_name].std(ddof=0)
av_part[m_name] = (av_part[m_name] - av_part[m_name].mean())/av_part[m_name].std(ddof=0)
plt.close('all')
sns.regplot(y=m_name, x=c_name, data=av_part)
sp = abs(av_part[[c_name, m_name]].corr('spearman').iloc[0,1])
plt.text(1,1,f'Spearman R {np.round(sp,2)}')
plt.show()

#%% regressor on evoked data. i.e. does offset predict activity in maintenance phase
good_ids = list(av_part.id) # ids we have data for
cue_files = [f for f in listdir(epodir) if 'postcue' in f] # cue files
_id = good_ids[0]
name = [''
def get_evoked(_id, name):
    try:
        file = [i for i in cue_files if _id in i][0]
        epochs = mne.epochs.read_epochs(join(epodir, file))
        #load in meta data to epochs
        evoked = epochs[name].average()
        return evoked
    except:
        return None

left = joblib.Parallel(n_jobs=5)(
    joblib.delayed(get_evoked)(_id, 'L_CUE') for _id in good_ids)

#%%


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
    joblib.delayed(decode_probe)(_id) for _id in good_ids)

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
cue_files = [f for f in listdir(epodir) if 'wholecue' in f]
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


#%% event epoch

cleandir = join(constants.BASE_DIRECTORY, 'cleaned') # dire
clean = [f for f in listdir(cleandir) if 'no' not in f]
ids = list(set([i.split('_')[0] for i in clean]))
ids.sort()

thisind = 3

all = [[i for i in clean if ii in i] for ii in good_ids]
this_raw = all[thisind]
this_id = good_ids[thisind]
this_trials = all_trials[all_trials.id == this_id]
#this_trials = this_trials[this_trials.prac != 1]


raw = mne.io.read_raw_fif(join(cleandir, this_raw[0]), preload=True)

#event_dict = {'Start': 201}
event_dict = {'L_CUE': 250,'R_CUE': 251,'N_CUE': 252}
time_dict = {'tmin': 0.0,'tmax': 1.5,'baseline': None}

events = mne.find_events(raw, shortest_event=1)
epochs = mne.Epochs(raw, events, event_dict, tmin=time_dict['tmin'],
                    tmax=time_dict['tmax'], baseline=time_dict['baseline'],
                    preload=True, proj=True)


# get names of cue epochs
c_epochs = mne.Epochs(raw, events, {'L_CUE': 250,'R_CUE': 251,'N_CUE': 252}, tmin=time_dict['tmin'],
                    tmax=time_dict['tmax'], baseline=time_dict['baseline'],
                    preload=True, proj=True)

# match to cue_dir column in all trials
names = [c_epochs[i]._name for i in range(len(c_epochs))]
names = [i.replace('N_CUE', '0') for i in names]
names = [i.replace('L_CUE', '-1') for i in names]
names = [i.replace('R_CUE', '1') for i in names]
names = [int(i) for i in names]

# get that column as list
cue_col = list(this_trials.cue_dir)

a = cue_col
b = names
[(i, i+len(b)) for i in range(len(a)) if a[i:i+len(b)] == b]


#match to behavioural data

# find logging start (in date string)
log_date = behav_starts[behav_starts.name == f'{this_id}_trials.txt'].time.iloc[0]
log_date = pd.to_datetime(log_date, unit='ns')

# find first event
b_zero = this_trials.iloc[0].iti_onset
date_b_zero =  log_date + datetime.timedelta(milliseconds=b_zero)

#get timestamp for first recording
date_meg_zero = pd.to_datetime(epochs.info['meas_date'])

# sometimes the first event doesn't come through, so we need to adjust our zero-time for this!
acceptable_events = [201,202,203,250,251,252,205,240,241] # first events we will accept
e_names = ['ITI', 'STIM', 'DELAY', 'LEFT_C', 'R_CUE', 'N_CUE', 'POSTCUE', 'PROBE_L', 'PROBE_R'] # names of events
first_ITI = this_trials.iloc[0].iti # ITI of the first trial
adjustments =[0, first_ITI, first_ITI+1000, first_ITI+2000, first_ITI+2000, first_ITI+2000, first_ITI+2500,
              first_ITI+3500,first_ITI+3500,] # adjustments for these events
# same for MEG events
# now find first acceptable event index
first = [(i, val) for i, val in enumerate(events[:,2]) if val in acceptable_events][0]
first_t = events[first[0],0] # get time stamp
t_zero = first_t - (adjustments[acceptable_events.index(first[1])]) # adjust according to adjustments


# for every trial work out an encapsulating time range zero'd to the first ITI
this_trials['z_start'] = this_trials['iti_onset'] - b_zero
this_trials['z_end'] = this_trials['probe_onset'] - b_zero
this_trials['z_end'] = this_trials['z_end'] + this_trials['resp_onset'] + this_trials['resp_duration']

# Do the same for absolute date
this_trials['date_start'] = log_date + pd.Series([pd.Timedelta(milliseconds=i) for i in this_trials.iti_onset], index=this_trials.index)
this_trials['date_end'] = this_trials['probe_onset']+ this_trials['resp_onset'] + this_trials['resp_duration']
this_trials['date_end'] = log_date + pd.Series([pd.Timedelta(milliseconds=(i)) for i in this_trials['date_end']], index=this_trials.index)

this_trials['end'] = this_trials['probe_onset'] + this_trials['resp_onset'] + this_trials['resp_duration']
this_trials['start'] = this_trials['iti_onset']
# loop through epochs and find matching meta data
# meta_df = pd.DataFrame(columns=this_trials.columns)
# unfound_ind = []
# for ind in range(len(epochs)):
#     this_ztime = epochs[ind].events[0,0] - t_zero
#     if this_ztime < 0:
#         t_zero = epochs[ind].events[0,0]
#         this_ztime = epochs[ind].events[0, 0] - t_zero
#     mask = (this_trials['z_start'] <= this_ztime) & (this_trials['z_end'] >= this_ztime)
#     if mask.sum() == 0:
#         unfound_ind.append(ind)
#     meta_df = meta_df.append(this_trials[mask], ignore_index=True)

#%% plot out ranges

trial_adjust = 2.3*60000
import copy
#%%
#trial_adjust = 24*60000

trial_adjust = 0
#trial_adjust = 11*60000
epoch = copy.copy(epo)
epoch_points = [epoch[i].events[0,0] for i in range(len(epoch))]
trial_ranges = [(b-trial_adjust, e-trial_adjust) for b, e in zip(this_trials.iti_onset, this_trials.probe_onset)]

fig, ax = plt.subplots()
ax.plot( epoch_points,[1.75]*len(epoch_points),'+') # plot epoch points
for t in trial_ranges: # plot ranges for each trial
    x_values = [t[0], t[1]]
    y_values = [2, 2]
    ax.plot(x_values, y_values)

#ax.set_xlim(1*60000,6*60000)
ax.set_ylim(0,3)
plt.show()
#%%
event_dict = {'L_CUE': 250, 'R_CUE': 251, 'N_CUE': 252}
time_dict = {'tmin': 0.0, 'tmax': 1.5, 'baseline': None}
thisid = '99005'
this_files = [i for i in listdir(cleandir) if thisid in i]


epolist = []
datelist = []
for file in this_files:
    raw = mne.io.read_raw_fif(join(cleandir, file),preload=True)
    datelist.append(raw.info['meas_date'])
    events = mne.find_events(raw, shortest_event=1)
    events = find_events_CBU(raw)
    epochs = mne.Epochs(raw, events, event_dict, tmin=time_dict['tmin'],
                        tmax=time_dict['tmax'], baseline=time_dict['baseline'],
                        preload=True, proj=True)
    epolist.append(epochs)

if len(this_files) > 1:
    time_diff = datelist[1].timestamp() - datelist[0].timestamp()
    for i in range(len(epolist[1])):
        epolist[1].events[i,0] = epolist[1].events[i,0] + (time_diff*1000)
    epo = mne.concatenate_epochs(epolist, add_offset=False)
else:
    epo = epolist[0]

#%%
this_trials = all_trials[all_trials.id == thisid]
this_trials['end'] = this_trials['probe_onset'] + this_trials['resp_onset'] + this_trials['resp_duration']
this_trials['start'] = this_trials['iti_onset']

# do the same as above but use timestamps instead!
meta_df = pd.DataFrame(columns=this_trials.columns)
unfound_ind = []
timings = []

aligned = False
trial_adjust = 0
iterations = 0
itercrement = 100
iterlim = 10000
time_array = this_trials[['start', 'end']].to_numpy() # array is faster
epo_ev = epo.events.copy()
while not aligned:
    print(iterations)
    unfound_ind = []
    trial_inds = []
    for ind in range(len(epo)):
        epo_s = epo_ev[ind,0] # this epoch's time in ms
        mask = (time_array[:,0]-trial_adjust <= epo_s) & (time_array[:,1]-trial_adjust >= epo_s)
        if mask.sum() == 0:
            unfound_ind.append(ind)
        else:
            trial_inds.append(np.where(mask==True)[0][0])
    if len(unfound_ind) == 0:
        meta_df = this_trials.iloc[trial_inds]
        epo.metadata = meta_df
        aligned = True
    else:
        trial_adjust = trial_adjust + 250
        iterations +=1
        if iterations > iterlim:
            break
#%%
print(epo['N_CUE'].metadata.cue_dir)
