# Hierarchical Bayesian Model for LPS Brain Scores/Clusters

# Updated 2024-08-08
# Created 2024-08-06

# DESCRIPTION
  # model interoception ~ exteroception cluster scores (and/or brain scores) with repeated measures (blocks)

# NEXT: troubleshooting "ConvergenceWarning: Maximum Liklihood optimization failed to converge"
  # try making conditions 0 and 1 (integers instead of pandas objects)
  # input needs to be one df with a column of the brainscores for intero, and another for extero

# %% Load python pkgs and set paths
import os
import subprocess
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import re


# Set paths
# wkdir = f'/home/common/piaf/LPS_STHLM/analysis_2023/PLS/int_rs_denoised_newconds/1_ROIS/all_rois_n8_NVoxels2357/meanInteroExtero_LPS_pls_brainscores_models' # working dir/results dir (must have wrx permissions)
spat_dir = f'/home/common/piaf/LPS_STHLM/analysis_2023/PLS/int_rs_denoised_newconds/1_ROIS/all_rois_n8_NVoxels2357/rmoutliers_20conds_meanInteroExtero_lps_all_pls_1_n22' # dir of the matlab SPAT results
scores_csv = f'{spat_dir}/rmoutliers_20conds_meanInteroExtero_lps_all_pls_1_n22_extracted_mean_values.csv' # full path to brain scores/clusters/LVs CSV file
savedir = '/home/common/piaf/LPS_STHLM/analysis_2023/PLS/mixed_effects_models_brainscores/lps_intero-extero_allblocks'

# Create results directory
if os.path.isdir(savedir)==False:
  cmd = f'mkdir {savedir}'
  subprocess.run(cmd, shell=True, executable='/bin/bash')

# Load data
df_init = pd.read_csv(scores_csv, sep=',', header=0)
df_init.set_index('filename', inplace=True)

# Define conditions as integers
interoception = 0
exteroception = 1



# %% Reshape the df_init to include only brain scores or clusters; leave out paths and groups (groups column is all ones)

# df_reshape = pd.melt(df.reset_index(), id_vars=['filename'], var_name='condition_block', value_name='value') # value can either be cluster, brain score, or LV
cluster_cols = [col for col in df_init.columns if 'thp2n2_cluster' in col] # list cluster column names
brainscore_cols = [col for col in df_init.columns if 'bs_' in col] # list brain score column names
df_reshape = pd.concat([df_init['subjID'], df_init['condition'], df_init[cluster_cols], df_init[brainscore_cols]], axis=1).reset_index(drop=True) # create new df with only paths, conditions, and cluster values
df_reshape.rename(columns={'condition': 'condition_block'}, inplace=True) # rename column headers

# Extract condition and block using regex
def parse_condition_block(condition_block):
  match = re.match(r'mean_(\w+)_block(\d+)_lps', condition_block)
  if match:
      condition = match.group(1)
      block = int(match.group(2))
      # print(f"{condition}, {block}") # TEST
      return condition, block
  else:
    return None, None
    
# Separate condition and block into their own columns
df_reshape[['condition', 'block']] = df_reshape['condition_block'].apply(lambda x: pd.Series(parse_condition_block(x))) # apply function to separate cond and block
df_reshape = df_reshape.drop('condition_block', axis=1) # now condition and block have their own columns so we can remove this one
df_reshape = df_reshape[['subjID', 'condition', 'block'] + [ c for c in df_reshape if c not in ['subjID', 'condition', 'block']]] # move condition and block cols to left

# Replace condition names (pandas objects) with integers 0 and 1
df_reshape.replace('interoception', interoception, inplace=True)
df_reshape.replace('exteroception', exteroception, inplace=True)


# %%
# cluster_cols = [col for col in df_init.columns if 'thp2n2_cluster' in col] # list cluster column names
# brainscore_cols = [col for col in df_init.columns if 'bs_lv1' == col] # list brain score column names
lv_data = 'bs_lv1'

# for lv_data in [cluster_cols[0]]:
# for lv_data in brainscore_cols:
  # df = pd.concat([df_reshape['subjID'], df_reshape['condition'], df_reshape['block'], df_reshape[cluster]], axis=1)
df = pd.concat([df_reshape['subjID'], df_reshape['condition'], df_reshape['block'], df_reshape[lv_data]], axis=1)


# df_LV = pd.concat(df_reshape['subjID'],df_reshape['condition'], df_reshape['block'], df_reshape[''])
# df_intero = df_reshape[df_reshape['condition']=='interoception'] # just interoception rows

# Model cluster values as a function of 'condition' with random effects for 'subject' and 'block'
# model = smf.mixedlm(f"{lv_data} ~ condition", df, groups=df["block"], re_formula="~subjID") # , groups=df["subjID"], re_formula="~block"
model = smf.mixedlm("bs_lv1 ~ condition", df, groups=df["subjID"], re_formula="~block")
result = model.fit()

# Print the summary of the model
print(result.summary())

# import pymc3 as pm

# with pm.Model() as model:
#     # Priors for unknown model parameters
#     intercept = pm.Normal('intercept', mu=0, sigma=10)
#     slope = pm.Normal('slope', mu=0, sigma=10)
    
#     # Random effect for subjects
#     subj_effect = pm.Normal('subj_effect', mu=0, sigma=10, shape=len(df['subjID'].unique()))
    
#     # Linear model
#     mu = intercept + slope * df['condition'] + subj_effect[df['subjID']]
    
#     # Likelihood
#     sigma = pm.HalfNormal('sigma', sigma=1)
#     y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=df['bs_lv1'])
    
#     # Inference
#     trace = pm.sample(1000, return_inferencedata=True)
