"""
constants.py
Alex Anwyl-Irvine 2021

Specify all of your local variables (paths to data and configuration information)!

"""

from os.path import join

# This is your path to the main data, it must contain these directories:
## 'raw'
## 'MaxFiltered'
## 'cleaned'
## 'epoched'
## 'b_scripts'
## 'b_logs'
#
BASE_DIRECTORY = join('/imaging', 'astle', 'users', 'ai05', 'RED', 'RED_MEG', 'worms')

# List of participants we want to include in analysis
PARTICIPANTS = ['99107', '99014', '99091', '99016', '99068', '99013', '99003',
       '99106', '99007', '99128', '99114', '99083', '99076', '99112',
       '99005', '99140', '99035', '99142', '99115', '99012', '99051',
       '99026', '99028', '99011', '99078', '99045', '99073', '99144',
       '99053', '99118', '99002', '99090', '99056', '99146', '99034',
       '99030', '99060', '99044', '99020', '99125', '99031', '99141',
       '99119', '99102', '99124', '99080', '99037', '99025', '99086',
       '99027', '99023', '99096', '99009', '99063', '99070', '99018',
       '99019', '99139', '99015', '99004', '99099', '99021', '99134',
       '99040', '99010', '99058', '99066', '99072', '99029', '99057',
       '99064', '99145', '99059', '99101', '99048', '99089', '99092',
       '99047', '99071', '99067', '99038']

# Path to a CSV containing our meta data
META_PATH = join('/imaging', 'ai05', 'RED', 'RED_MEG', 'worms', 'Combined3.csv')

# This is your path to the remote python environment for sbatch commands
PYTHON_PATH  = join('/home/ai05/.conda/envs/mne_2/bin/python')

# This is the path that contains RED tools for sbatch imports
REPO_PATH = join('/home/ai05/WM_Worms')

TRIGGERS = {
       'ITI': 201,
       'STIM': 202,
       'DELAY': 203,
       'L_CUE': 250,
       'R_CUE': 251,
       'N_CUE': 252,
       'POSTCUE': 205,
       'L_PROBE': 240,
       'R_PROBE': 241,
       'RESP': 207,
       'FEEDBACK': 208
}

