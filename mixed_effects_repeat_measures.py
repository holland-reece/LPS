# Mixed Effects Linear Model with Repeated Measures for LPS Brain Scores

# Updated 2024-08-06
# Created 2024-08-06

# %% Load python pkgs and set paths
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import pearsonr


# Set paths
home = f'/home/common/turing/VOX_blood' # home dir
clin_csv = f'{home}/vox_clinical.csv' # clinical scales
blood_csv = f'{home}/vox_blood.csv' # blood measures

# Load data
clin_df = pd.read_csv(clin_csv, sep=',', header=0, index_col=0)
clin_df.drop(columns=['Unnamed: 17','0=Screening'], inplace=True) # remove cols without data

blood_df = pd.read_csv(blood_csv, sep=',', header=0)
# print(blood_df.head())
# df_blood = bloodstats(blood_df) # init blood stats class

# NOTE: only SAD group has data for the last two timepoints ('POST' after Tx, and '2Y-FU' 2-year followup)
timepoints = ['B1','B2','POST','2Y-FU']

blooddict = {
    'LPK': ([3.5,8.8],'10E9/L'), 'EPK': ([3.9,5.2],'10E9/L'), 'HB': ([117,153],'g/L'), 'EVF': ([0.35,0.46],''), 'MCH': ([27,32],'pg'), 'BMCV': ([82,98],'fL'), 
    'TPK': ([165,387],'10E9/L'), 'BLYMF': ([1.0,3.5],'10E9/L'), 'BEOS': ([0.07,0.3],'10E9/L'), 'BMONO': ([0.3,1.2],'10E9/L'), 'BNEUT': ([1.8,6.3],'10E9/L'),
      'BBASO': ([0.0,0.1],'10E9/L'), 'BMCHC': ([330,360],'g/L')
      }

