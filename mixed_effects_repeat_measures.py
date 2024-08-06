# Mixed Effects Linear Model with Repeated Measures for LPS Brain Scores

# Updated 2024-08-06
# Created 2024-08-06

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
wkdir = f'/home/common/piaf/LPS_STHLM/analysis_2023/PLS/int_rs_denoised_newconds/1_ROIS/all_rois_n8_NVoxels2357/meanInteroExtero_LPS_pls_brainscores_models' # working dir/results dir (must have wrx permissions)
spat_dir = f'/home/common/piaf/LPS_STHLM/analysis_2023/PLS/int_rs_denoised_newconds/1_ROIS/all_rois_n8_NVoxels2357/rmoutliers_20conds_meanInteroExtero_lps_all_pls_1_n22' # dir of the matlab SPAT results
scores_csv = f'{spat_dir}/rmoutliers_20conds_meanInteroExtero_lps_all_pls_1_n22_extracted_mean_values.csv' # full path to brain scores/clusters/LVs CSV file

# Create results directory
if os.path.isdir(wkdir)==False:
  cmd = f'mkdir {wkdir}'
  subprocess.run(cmd, shell=True, executable='/bin/bash')

# Load data
df = pd.read_csv(scores_csv, sep=',', header=0)
df.set_index('filename', inplace=True)

# Reshape the df to long format
df_long = pd.melt(df.reset_index(), id_vars=['filename'], var_name='condition_block', value_name='value') # value can either be cluster, brain score, or LV

# Extract condition and block using regex
def parse_condition_block(condition_block):
    match = re.match(r'mean_(\w+)_block(\d+)_lps', condition_block)
    if match:
        condition = match.group(1)
        block = int(match.group(2))
        cmd = f'echo -e "{condition}, {block}"'
        subprocess.run(cmd, shell=True, executable='/bin/bash')
        return condition, block
    return None, None

df_long[['condition', 'block']] = df_long['condition_block'].apply(lambda x: pd.Series(parse_condition_block(x)))

# # Fit the mixed effects model
# # Here, we model 'value' as a function of 'condition' with random effects for 'subject' and 'block'
# model = smf.mixedlm("value ~ condition", df_long, groups=df_long["subject"], re_formula="~block")
# result = model.fit()

# # Print the summary of the model
# print(result.summary())