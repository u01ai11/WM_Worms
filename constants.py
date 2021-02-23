"""
constants.py
Alex Anwyl-Irvine 2021

Specify all of your local variables (paths to data and configuration information)!

"""

from os.path import join

# This is your path to the main data, it must contain:
## 'raw'
## 'MaxFiltered'
## 'cleaned'
## 'epoched'
## 'b_scripts'
## 'b_logs'

BASE_DIRECTORY = join('imaging', 'ai05', 'RED', 'RED_MEG', 'worms')

# This is your path to the remote python environment for sbatch commands
PYTHON_PATH  = join('home/ai05/.conda/envs/mne_2/bin/python')

# This is the path that contains RED tools for sbatch imports
REPO_PATH = join('/home/ai05/WM_WORMS')