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

# %% Reshape the df to long format

# FIX: need to model 

# df_cluster = pd.melt(df.reset_index(), id_vars=['filename'], var_name='condition_block', value_name='value') # value can either be cluster, brain score, or LV
cluster_cols = [col for col in df.columns if 'thp2n2_cluster' in col] # list cluster column names
df_cluster = pd.concat([df['subjID'], df['path'], df['condition'], df[cluster_cols]], axis=1) # create new df with only paths, conditions, and cluster values
df_cluster.rename(columns={'condition': 'condition_block'}, inplace=True) # rename column headers

# Extract condition and block using regex
def parse_condition_block(condition_block):
  match = re.match(r'mean_(\w+)_block(\d+)_lps', condition_block)
  if match:
      condition = match.group(1)
      block = int(match.group(2))
      # print(f"{condition}, {block}")
      return condition, block
  else:
    return None, None
    
# Separate condition and block into their own columns
df_cluster[['condition', 'block']] = df_cluster['condition_block'].apply(lambda x: pd.Series(parse_condition_block(x)))

for cluster in [cluster_cols[0]]:
  # df_thiscluster = pd.concat([df_cluster['subjID'], df_cluster['condition'], df_cluster['block'], df_cluster[cluster]], axis=1)
  df_thiscluster = pd.concat([df_cluster['subjID'], df_cluster['condition'], df_cluster['block'], df_cluster[cluster]], axis=1).reset_index(drop=True)
 
 
  # df_LV = pd.concat(df_cluster['subjID'],df_cluster['condition'], df_cluster['block'], df_cluster[''])
  # df_intero = df_cluster[df_cluster['condition']=='interoception'] # just interoception rows

  # Model cluster values as a function of 'condition' with random effects for 'subject' and 'block'
  model = smf.mixedlm(f"{cluster} ~ condition", df_thiscluster, groups=df_thiscluster["block"], re_formula="~subjID") # , groups=df_thiscluster["subjID"], re_formula="~block"
  result = model.fit()

  # Extract fitted values for plotting
  # df_thiscluster['fitted'] = result.fittedvalues

  # Plot the results
  # plt.plot(df_thiscluster['block'], df_thiscluster['fitted'], label=cluster)
  plt.plot(result.fittedvalues)

# Add labels and legend
# plt.xlabel('Block')
plt.ylabel('Fitted Values')
plt.title(f'Cluster ({cluster}) values by condition (extero/interoception)')
# plt.legend(loc='best')

# Save and show the plot
plt.savefig(os.path.join(wkdir, 'fitted_values_by_cluster.png'))
plt.show()

# Print the summary of the model
# print(result.summary())
# %%
