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
