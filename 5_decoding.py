"""
5_decoding.py
Alex Anwyl-Irvine 2021

Within each participant:
- Train model on probes
- Classify the cues (where no probes are on screen)

"""

from os.path import join
from os import listdir
import os
import numpy as np
import mne

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
import seaborn as sns
import matplotlib.pyplot as plt

import sails
try:
    import constants
    from REDTools import preprocess
except:
    import sys
    sys.path.insert(0, '/home/ai05/WM_Worms')
    import constants
    from REDTools import preprocess


#%% Load in
#%% initial processing
cleandir = join(constants.BASE_DIRECTORY, 'cleaned') # dire
clean = [f for f in listdir(cleandir) if 'no' not in f]
ids = list(set([i.split('_')[0] for i in clean]))
ids.sort()
i = 1
part = ids[i]
part_files = [i for i in clean if part in i]

raw = mne.io.read_raw_fif(join(cleandir, part_files[0]), preload=True)
raw._first_samps[0] = 0
# Apply band-pass filter
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

# detect blinks
eog_events = mne.preprocessing.find_eog_events(raw)
n_blinks = len(eog_events)
# Center to cover the whole blink with full duration of 0.5s:
b_onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
b_duration = np.repeat(0.5, n_blinks)

# onset = np.concatenate([a_onset, b_onset])
# duration = np.concatenate([a_duration, b_duration])
# labels = ['artefacts'] * len(a_duration) + ['bad blink'] * n_blinks

onset = b_onset
duration = b_duration
labels = ['bad blink'] * n_blinks

# raw.annotations = mne.Annotations(onset, duration, labels,orig_time=raw.info['meas_date'])
raw.set_annotations(mne.Annotations(onset, duration, labels, orig_time=raw.info['meas_date']))

#%% get events
events = mne.find_events(raw, min_duration=0.002)
timings = [(-0.1, 1), (-0.1, 1), (-0.1, 5), (-0.1, 5)]
tnames = ['STIM', 'POSTCUE', 'L_PROBE', 'R_PROBE']
# extract epochs
epo_list = [mne.Epochs(raw, events, event_id=constants.TRIGGERS[n], tmin=t[0], tmax=t[1]) for n, t in zip(tnames, timings)]
# downsample
[i.load_data() for i in epo_list]
[i.resample(200., npad='auto') for i in epo_list]

#evokeds = [i.average() for i in[epochs_lprobe, epochs_rprobe]]
evokeds = [epo_list[0].average()]
mne.viz.plot_evoked_topo(evokeds)

#%% Try MVPA
events = mne.find_events(raw, min_duration=0.002)

def process_epoch(raw, events, tdict, times, baseline):
    epochs = mne.Epochs(raw, events, tdict, tmin=times[0], tmax=times[1], baseline=baseline,
                          preload=True, proj=True)
    epochs.pick_types(meg=True, exclude='bads')
    epochs.resample(200., npad='auto')
    return epochs
#%%
#data
# left and right probe
p_epochs = process_epoch(raw, events, {'L_PROBE': 240,'R_PROBE': 241},
                         times=(0,1), baseline=(0, 0.1))
# postcue
cp_epochs = process_epoch(raw, events, {'POSTCUE': 205},
                          times=(0,1), baseline=(0, 0.1))
#cues but from the offset
c_epochs = process_epoch(raw, events, {'L_CUE': 250,'R_CUE': 251,'N_CUE': 252},
                         times=(0.5,1.5), baseline=(0.5, 0.6))


#%% first look at ERPs
c_epochs.apply_baseline((0.5,0.6))
cevoked_list = [c_epochs[i].average() for i in ['L_CUE', 'R_CUE', 'N_CUE']]
mne.viz.plot_compare_evokeds(cevoked_list)

pevoked_list = [p_epochs[i].average() for i in ['L_PROBE', 'R_PROBE']]
mne.viz.plot_compare_evokeds(pevoked_list)
#%% Now see if we can decode the probes (should be easy)
p_epochs = process_epoch(raw, events, {'L_PROBE': 240,'R_PROBE': 241},
                         times=(0,1), baseline=None)
X = p_epochs.get_data()  # MEG signals: n_epochs, n_meg_channels, n_times
y = p_epochs.events[:, 2]  # target: left vs right probes

clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))
time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc', verbose=True)
scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=5)
scores = np.mean(scores, axis=0)

sns.lineplot(p_epochs.times, scores)
plt.show()
#%% CLF
#Train it on our probes

scaler = StandardScaler()

p_epochs = process_epoch(raw, events, {'L_PROBE': 240,'R_PROBE': 241},
                         times=(0,1), baseline=None)
cp_epochs = process_epoch(raw, events,  {'L_CUE': 250,'R_CUE': 251,'N_CUE': 252},
                          times=(0.5,1.5), baseline=None)
# only take l/r/ cues
e = process_epoch(raw, events,  {'L_CUE': 250,'R_CUE': 251},times=(0.5,1.5), baseline=None)
#%%
model = LogisticRegression(penalty='elasticnet', solver='saga',l1_ratio=1)
#model = RidgeClassifier()
X = p_epochs.copy().get_data()
#X = X.reshape(len(X),-1)
X = scaler.fit_transform(X)
y = p_epochs.events[:, 2]
model.fit(X, y)

# now predict each postcue epoch
X_test = cp_epochs.get_data()
X_test = X_test.reshape(len(X_test),-1)
X_test = scaler.fit_transform(X_test)
y_pred = model.predict_proba(X_test)


y_t = e.events[:, 2]
y_t = np.array([240 if i == 250 else 241 for i in y_t]) # relabel to match
X_t = e.get_data()
X_t = X_t.reshape(len(X_t),-1)
X_t = scaler.fit_transform(X_t)
score = model.score(X_t, y_t)

#Label predictions
events_labels = cp_epochs.events[:,2]
labelled_preds = np.zeros((y_pred.shape[0],y_pred.shape[1]+1))
labelled_preds[:,:2] = y_pred
labelled_preds[:,2] = events_labels
#
# #confusion matrix
confusion = np.zeros((2, 3)) # p_left, p_right x c_left, c_right, c_neutral

for ii, train_class in enumerate([240, 241]): #left probe and right probe
    for jj, test_class in enumerate([250, 251, 252]): #left cue and right cue and neutral cue
        confusion[ii, jj] = np.mean(labelled_preds[labelled_preds[:,2] == test_class,ii])
print(confusion)

#%% Try again but with different simpler model

# Here we use a logistic regression for an RSA type analusis
# create a linear model with LogisticRegression
clf = LogisticRegression(solver='lbfgs')
scaler = StandardScaler()
model = LinearModel(clf)

labels = p_epochs.events[:, -1]

# get MEG and EEG data
meg_epochs = p_epochs.copy().pick_types(meg=True, eeg=False)
meg_data = meg_epochs.get_data().reshape(len(labels), -1)

# fit the classifier on MEG data
X = scaler.fit_transform(meg_data)
model.fit(X, labels)

left_test = cp_epochs['L_CUE'].get_data().reshape(len(cp_epochs['L_CUE'].events), -1)
left_test = scaler.fit_transform(left_test)
left_pred = model.predict_proba(left_test)

right_test = cp_epochs['R_CUE'].get_data().reshape(len(cp_epochs['R_CUE'].events), -1)
right_test = scaler.fit_transform(right_test)
right_pred = model.predict_proba(right_test)