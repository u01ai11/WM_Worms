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
import glmtools
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

all_trials.to_csv(join(constants.BASE_DIRECTORY, 'all_trials.csv'))

#%% load meta
meta = pd.read_csv(join(constants.BASE_DIRECTORY, 'Combined3.csv'))

#%%
"""
1. ERROR FOR CUED AND UNCUED TRIALS 
"""

def precision(_list):
    return 1/ss.circstd(np.deg2rad(_list), low=0, high= np.deg2rad(180))

def robust_mean(_list):
    data = np.array(copy.deepcopy(_list))
    data[data == 0] = np.nan
    data[np.abs(data - data.mean()) > 3 * data.std()] = np.nan
    return np.nanmean(data)

#1.1 Calculate participantwise Accuracy and Precision
#Accuracy: reciprocal of the circulat standard deviation - i.e. vpn misses
group_r = []
for _id in all_trials.id.unique():
    print(_id)
    _trials = all_trials[all_trials.id ==_id]
    _trials = _trials[_trials.prac == 0]
    _trials = _trials.iloc[0:60]

    # Precision
    _prec_all = precision(_trials.offset.abs())
    _prec_v = precision(_trials[_trials.cue_type == 'valid'].offset.abs().to_list())
    _prec_n = precision(_trials[_trials.cue_type == 'neutral'].offset.abs().to_list())
    _prec_l = precision(_trials[_trials.cue_dir == 0].offset.abs().to_list())
    _prec_r = precision(_trials[_trials.cue_dir == 1].offset.abs().to_list())

    # Error
    _acc_all = np.mean(np.deg2rad(_trials.ang_dist.abs().to_list()))
    _acc_v = np.deg2rad(np.mean(np.abs(_trials[_trials.cue_type == 'valid'].ang_dist)))
    _acc_n = np.deg2rad(np.mean(np.abs(_trials[_trials.cue_type == 'neutral'].ang_dist)))
    _acc_l = np.deg2rad(np.mean(np.abs(_trials[_trials.cue_dir == 0].ang_dist)))
    _acc_r = np.deg2rad(np.mean(np.abs(_trials[_trials.cue_dir == 1].ang_dist)))

    #Onset Time
    _onset_all = robust_mean(_trials.resp_onset)
    _onset_n =robust_mean(_trials[_trials.cue_type == 'neutral'].resp_onset)
    _onset_l = robust_mean(_trials[_trials.cue_dir == 0].resp_onset)
    _onset_r = robust_mean(_trials[_trials.cue_dir == 1].resp_onset)

    #Duration Time
    _duration_all = robust_mean(_trials.resp_duration)
    _duration_n =robust_mean(_trials[_trials.cue_type == 'neutral'].resp_duration)
    _duration_l = robust_mean(_trials[_trials.cue_dir == 0].resp_duration)
    _duration_r = robust_mean(_trials[_trials.cue_dir == 1].resp_duration)
    group_r.append((_acc_v, _acc_n, _acc_l, _acc_r, _acc_all, _prec_v, _prec_n, _prec_l, _prec_r, _prec_all,
                    _onset_all, _onset_n, _onset_l, _onset_r, _duration_all, _duration_n, _duration_l, _duration_r, float(_id)))
#%% create combined spreadsheet

comb = pd.DataFrame(data=np.array(group_r), columns=['Error Valid', 'Error Neutral', 'Error Left', 'Error Right', 'Overall Error', 'Precision Valid',
                                                     'Precision Neutral', 'Precision Left', 'Precision Right', 'Overall Precision',
                                                     'Overall Onset', 'Onset Neutral', 'Onset Left', 'Onset Right',
                                                     'Overall Duration', 'Duration Neutral', 'Duration Left', 'Duration Right', 'id'])
comb['id'] = [str(int(i)) for i in comb['id']]
comb['Alex_ID'] = [str(i) for i in comb['id']]
meta['Alex_ID'] = [str(i) for i in meta['Alex_ID']]
comb = comb.merge(meta, on='Alex_ID')
#%%
comb.to_csv(join(constants.BASE_DIRECTORY, 'worms_meta.csv'))

directions = ['Neutral', 'Left', 'Right']
for i, name in enumerate(['Error', 'Precision', 'Onset', 'Duration']):
    tmp = pd.melt(comb, id_vars=['Alex_ID'], value_vars=[f'{name} {direct}' for direct in directions])
    tmp['Cue'] = [ii.split(' ')[1] for ii in tmp['variable']]
    tmp[name] = tmp['value']
    tmp = tmp.drop(columns=['value', 'variable'])
    if i == 0:
        melted = copy.deepcopy(tmp)
    else:
        melted = pd.merge(melted, tmp, on=['Alex_ID', 'Cue'])
melted.to_csv(join(constants.BASE_DIRECTORY, 'manova_table.csv'))
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


#%% statistical annotation function
def stat_anno(data, offset, depth, color, ax, pos):
    fact = data.max() /100
    y, h, col = data.max() + fact*offset, fact*depth, color
    x1, x2 = pos
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)/2, y+(h/2), "*", ha='center', va='bottom', color=col)
#%% Left vs Right vs Nutral Cue
tmp = pd.melt(comb, value_vars=["Error Neutral", "Error Left", "Error Right"], id_vars=['id'])
fig, ax = plt.subplots(1,2)
sns.boxplot(x="variable", y="value", data=tmp, ax=ax[0])
sns.swarmplot(x="variable", y="value", data=tmp, ax=ax[0],color=".25")
ax[0].set_ylabel('Error (Radians)')
ax[0].set_xlabel('Cue Type')
ax[0].set_xticklabels(['Neutral', 'Left', 'Right'])
stat_anno(tmp["value"], 3, 2, 'k',ax[0], (0,2))
ax[0].text(-0.1, 1.1, 'A', transform=ax[0].transAxes,
        size=20, weight='bold')
tmp = pd.melt(comb, value_vars=["Precision Neutral", "Precision Left", "Precision Right"], id_vars=['id'])
sns.boxplot(x="variable", y="value", data=tmp, ax=ax[1])
sns.swarmplot(x="variable", y="value", data=tmp, ax=ax[1],color=".25")
ax[1].set_ylabel('Precision (1/SD)')
ax[1].set_xlabel('Cue Type')
ax[1].set_xticklabels(['Neutral', 'Left', 'Right'])
stat_anno(tmp["value"],  3, 2, 'k',ax[1], (1,2))
stat_anno(tmp["value"],  6, 2, 'k',ax[1], (0,2))
ax[1].text(-0.1, 1.1, 'B', transform=ax[1].transAxes,
        size=20, weight='bold')
plt.tight_layout()
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

#%%
"""
#########
Duration
   &
 Onset
######### 
"""

plt.close('all')
sns.set(style="whitegrid", font_scale=1.5, palette="Set2")
#sns.set()
tmp = pd.melt(comb, value_vars=["Overall Onset", "Overall Duration"])
g = sns.FacetGrid(tmp, col="variable", sharex=False, sharey=True, despine=True, height=6)
g.map(sns.histplot, "value", common_norm=True, kde=True)
g.map(sns.rugplot, "value")
g.set_axis_labels(y_var='Count')
g.axes[0][0].set_xlabel('Onset Time (ms)')
g.axes[0][1].set_xlabel('Duration Time (ms)')
g.set_titles(col_template="", row_template="")
plt.tight_layout()

#%%
#%% Left vs Right vs Nutral Cue
sns.set(style="whitegrid", font_scale=1.5, palette="Set3")
tmp = pd.melt(comb, value_vars=["Onset Neutral", "Onset Left", "Onset Right"], id_vars=['id'])
fig, ax = plt.subplots(1,2)
sns.boxplot(x="variable", y="value", data=tmp, ax=ax[0])
sns.swarmplot(x="variable", y="value", data=tmp, ax=ax[0],color=".25")
ax[0].set_ylabel('Onset Time (ms)')
ax[0].set_xlabel('Cue Type')
ax[0].set_xticklabels(['Neutral', 'Left', 'Right'])
stat_anno(tmp["value"], 3, 2, 'k',ax[0], (0,1))
stat_anno(tmp["value"],  6, 2, 'k',ax[0], (0,2))
ax[0].text(-0.1, 1.1, 'A', transform=ax[0].transAxes,
        size=20, weight='bold')
tmp = pd.melt(comb, value_vars=["Duration Neutral", "Duration Left", "Duration Right"], id_vars=['id'])
sns.boxplot(x="variable", y="value", data=tmp, ax=ax[1])
sns.swarmplot(x="variable", y="value", data=tmp, ax=ax[1],color=".25")
ax[1].set_ylabel('Duration Time (ms)')
ax[1].set_xlabel('Cue Type')
ax[1].set_xticklabels(['Neutral', 'Left', 'Right'])
ax[1].text(-0.1, 1.1, 'B', transform=ax[1].transAxes,
        size=20, weight='bold')
plt.tight_layout()

#%%
"""
##################
GLM - Covariates #
##################

Let's have a look at weather the above metrics relate to anything else - i.e. performance and cue effect

"""

#%% set up the glm data matrix -- array with one row per participant

# Specify and calculate our key outcomes for the GLM
outcome_vars = ['Overall Error', 'Overall Precision', 'Overall Onset', 'Overall Duration']
comb['Onset Valid'] = comb[['Onset Left','Onset Right']].mean(axis=1)
comb['Duration Valid'] = comb[['Duration Left','Duration Right']].mean(axis=1)
glm_df = comb.drop(columns=['AMY_ID', 'Dan_ID'], inplace=False)
glm_df = glm_df[glm_df.WASI_Mat > 20]
#drop rows with NaNs in covariate or data columns - threshold = 4
glm_df.dropna(thresh= 4, inplace=True)
#glm_df.dropna(inplace=True)
glm_df = glm_df.fillna(glm_df.median())
for metric in ['Onset', 'Duration', 'Precision', 'Error']:
    glm_df[f'Cue Effect {metric}'] = glm_df[f'{metric} Valid'] - glm_df[f'{metric} Neutral']
    outcome_vars.append(f'Cue Effect {metric}')

outcome_data = np.array(glm_df[outcome_vars])
out_labels = ['Average Error (Radians)', 'Average Precision (1/SD)', 'Average Onset (ms)', 'Average Duration (ms)', 'Error Cue Effect (Radians)',
              'Precision Cue Effect (Radians)', 'Onset Cue Effect (ms)', 'Duration Cue Effect (ms)']
dat = glmtools.data.TrialGLMData(data=outcome_data, dim_labels=['participants', 'metrics'])

# specify the predictors/covariates
cov_labels = ['Age (years)', 'WASI Matrix Reasoning', 'SDQ Score', 'Objective SES Score', 'Subjective SES Score', 'Household SES Score']
covariates = ['Age', 'WASI_Mat', 'SDQ_total', 'objective_SES', 'subjective_SES', 'household_SES']

regs = list()
regs.append(glmtools.regressors.ConstantRegressor(num_observations=len(glm_df)))
for i, name in enumerate(covariates):
    regs.append(glmtools.regressors.ParametricRegressor(values=np.array(glm_df[name]),
                                                        name=name,
                                                        preproc='z',
                                                        num_observations=len(glm_df)))
# contrasts
contrasts = list()
names = ['Intercept'] + covariates
for i, _reg in enumerate(regs):
    cont_vals = [0] * (len(covariates)+1)
    cont_vals[i] = 1
    contrasts.append(glmtools.design.Contrast(name=names[i], values=cont_vals))
#design matrix
des = glmtools.design.GLMDesign.initialise(regs,contrasts)

# model
#model = glmtools.fit.OLSModel( des, dat, standardise_data=False)
model = glmtools.fit.SKLModel(design=des, data_obj=dat, fit_args={'lm': 'LinearRegression', 'batch':'sklearn'}, standardise_data=True)
#%% permute the beta weights from the OLS regression
perms = 100000
sig_thresh = 95
sig_mask = []
CPs =[]
for i in range(1,len(regs)):
    CP = glmtools.permutations.Permutation(des, dat, i, perms, metric='betas', nprocesses=2)
    results = CP.get_sig_at_percentile(sig_thresh)
    sig_mask.append(results)
    CPs.append(CP)
sig_mask = np.array(sig_mask)

#%%
np.save('sig_mask.npy',sig_mask,  allow_pickle=True)
np.save('CPs.npy',CPs, allow_pickle=True)
#%% calculate regression line matrix at the regressor level
# i.e. the effect of each predictor on ALL outcome metrics
# This is tricky to interpret, because not all predictor-outcome relationships are significant after permuting
maxs=model.design_matrix.max(axis=0)
mins=model.design_matrix.min(axis=0)
xlines = np.zeros([10,model.design_matrix.shape[1]])
for i in range(model.design_matrix.shape[1]):
    xlines[:,i] = np.linspace(start=mins[i], stop=maxs[i], num=10)
ylines= model.skm.predict(xlines)

fig, ax = plt.subplots(model.design_matrix.shape[1])
for i in range(model.design_matrix.shape[1]):
    ax[i].plot(xlines[:,i], ylines[:,i])
#%%
#%% create a summary plot
#plt.close('all')

palette = itertools.cycle(sns.color_palette()) # for the pallete

total_sig = np.sum(sig_mask)
fig, ax = plt.subplots(2,4)
ax = ax.ravel()
#fig.delaxes(ax[7])
plti = 0

def z2raw(z, og):
    return np.mean(og) + (z*np.std(og))

for i, CP in enumerate(CPs):
    name = CP.cname
    sig_ind =[ii for ii, x in enumerate(CP.get_sig_at_percentile(sig_thresh)) if x]
    for ii, ind in enumerate(sig_ind):
        #x = np.array(glm_df[name])
        # X value is our predictor
        x = CP._design.design_matrix[:,i+1]
        # generate an equally spaced point line for the angle of regression (in normalised space)
        pr_x = np.linspace(start=x.min(), stop=x.max(), num=10)
        # unnormalise predictor
        x_raw = np.array([z2raw(iii, glm_df[CP.cname]) for iii in x])
        # unnormaise the line array
        pr_x_raw = np.array([z2raw(iii, glm_df[CP.cname]) for iii in pr_x])
        # Y will be our predicted outcome score
        y=ss.zscore(dat.data[:,ind])
        # the regression line for y is the normalised X values multiplied by the coefficent/beta
        pr_y = pr_x * model.betas[i+1, ind]
        #raw version of y
        y_raw = np.array([z2raw(iii, glm_df[outcome_vars[ind]]) for iii in y])
        # raw version of regression vector needs to use the transformed beta
        pr_y_raw = np.array([z2raw(iii, glm_df[outcome_vars[ind]]) for iii in pr_y])
        #sns.regplot(x,y, ax=ax[plti],truncate=True)
        sns.regplot(x_raw, y_raw, ax=ax[plti], truncate=True, fit_reg=False, color=next(palette))
        ax[plti].plot(pr_x_raw, pr_y_raw, color='black')
        ax[plti].set_xlabel(cov_labels[i])
        ax[plti].set_ylabel(outcome_vars[ind])
        ax[plti].set_title(f'Standardised Beta: {np.round(model.betas[i+1, ind],3)}')
        ax[plti].text(-0.1, 1.1, chr(ord('@')+(plti+1)), transform=ax[plti].transAxes,
                   size=20, weight='bold')
        plti+=1

plt.tight_layout()