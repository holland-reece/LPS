# Mixed Effects Linear Model with Repeated Measures: Intero- vs. Exteroception Mean-Centered PLS Brain Scores

# Updated 2024-09-19
# Created 2024-09-19

# DESCRIPTION
    # models interoception ~ exteroception cluster scores (and/or brain scores) with repeated measures (blocks)
    # plots subjects' brainscores (for one LV) over time (requires you to input error margins from PLS reports)

# %% Load python pkgs and set paths
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf # for glm/mixed-effects models
import re # for sorting values from the CSV

# for ANOVA
# import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

# to calculate 95% CI for plotting
from scipy import stats



# Set paths
spat_dir = f'/home/common/piaf/LPS_STHLM/analysis_2023/PLS/int_rs_denoised_newconds/1_ROIS/all_rois_n8_NVoxels2357/meanIntero_lps_10blocks_all_pls_1_n22' # dir of the matlab SPAT results
scores_csv = f'{spat_dir}/meanIntero_lps_10blocks_all_pls_1_n22_extracted_mean_values.csv' # full path to brain scores/clusters/LVs CSV file
savedir = '/home/common/piaf/LPS_STHLM/analysis_2023/PLS/mixed_effects_models_brainscores/lps_intero_allblocks'

# spat_dir = f'/home/common/piaf/LPS_STHLM/analysis_2023/PLS/int_rs_denoised_newconds/1_ROIS/all_rois_n8_NVoxels2357/rmoutliers_20conds_meanInteroExtero_lps_all_pls_1_n22' # dir of the matlab SPAT results
# scores_csv = f'{spat_dir}/rmoutliers_20conds_meanInteroExtero_lps_all_pls_1_n22_extracted_mean_values.csv' # full path to brain scores/clusters/LVs CSV file
# savedir = '/home/common/piaf/LPS_STHLM/analysis_2023/PLS/mixed_effects_models_brainscores/lps_intero-extero_allblocks'

y_error = np.array([2.2785, 3.8976]) # get from SPAT report BSR range

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
# brainscore_ci_up = [col for col in df_init.columns if 'ci_up' in col] # list brain score column names
# brainscore_ci_ll = [col for col in df_init.columns if 'ci_ll' in col] # list brain score column names

df_reshape = pd.concat([df_init['subjID'], df_init['condition'], df_init[cluster_cols], df_init[brainscore_cols]], axis=1).reset_index(drop=True) # create new df with only paths, conditions, and cluster values
df_reshape.rename(columns={'condition': 'condition_block'}, inplace=True) # rename column headers

# Extract condition and block using regex
def parse_condition_block(condition_block):
  match = re.match(r'mean_(\w+)_block(\d+)_lps', condition_block)
  # match = re.match(r'mean_(\w+)_block(\d+)_pl', condition_block)
  if match:
      condition = match.group(1)
      block = int(match.group(2))
      print(f"{condition}, {block}") # TEST
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

# %% Define and fit the model
# lv_data = 'bs_lv3' # specify brain score or latent variable you want to model
# lv_data_string = 'Brainscore LV3' # string for titles and plot headers and such
lv_data = 'lv3_thp2n2_cluster2'
lv_data_string = 'LV3 Cluster 2: Right Mid Cingulate (thr +/-2)'

# If brainscore, also extract ci_up and ci_ll (lower and upper CI bounds)
if 'bs' in lv_data:
    df = pd.concat([df_reshape['subjID'], df_reshape['condition'], df_reshape['block'], df_reshape[lv_data], df_reshape[f'{lv_data}_ci_up'], df_reshape[f'{lv_data}_ci_ll']], axis=1)
# If cluster, there are no CI bounds saved in the CSV
elif 'cluster' in lv_data:
    df = pd.concat([df_reshape['subjID'], df_reshape['condition'], df_reshape['block'], df_reshape[lv_data]], axis=1)

# Model cluster values as a function of 'condition' with random effects for 'subject' and 'block'
model = smf.mixedlm(f"{lv_data} ~ block", df, groups=df["subjID"]) # acknowledges that effect of block on brainscores ~ conditions may differ between subjects
# model = smf.mixedlm(f"{lv_data} ~ condition", df, groups=df["subjID"], re_formula="~block") # try with groups=df["block"] (averages across subjects?)
result = model.fit()

# Print the summary of the model
print(result.summary())
subprocess.run(f'echo "{result.summary()}" >> {savedir}/{lv_data}_model_summary.txt', shell=True, executable='/bin/bash')


# %% 1-way ANOVA

# lv_data = 'bs_lv3' # specify brain score or latent variable you want to model

# # Don't need condition here (only have one)
# df = pd.concat([df_reshape['subjID'], df_reshape['block'], df_reshape[lv_data]], axis=1)

# If using repeated measures ANOVA (AnovaRM) with subject as a random effect:
anova_rm = AnovaRM(df, depvar=lv_data, subject='subjID', within=['block'])
result = anova_rm.fit()

# Print the summary of the repeated measures ANOVA
print(result.summary())
subprocess.run(f'echo "{result.summary()}" >> {savedir}/{lv_data}_anova_summary.txt', shell=True, executable='/bin/bash')

# %% Plot actual vs. predicted from model results

# Add the predicted values to the DataFrame
df['predicted'] = result.fittedvalues

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(df[lv_data], df['predicted'])
plt.xlabel(f'Actual {lv_data}')
plt.ylabel(f'Predicted {lv_data}')
plt.title(f'Actual vs Predicted {lv_data}')
plt.plot([df[lv_data].min(), df[lv_data].max()],
         [df[lv_data].min(), df[lv_data].max()],
         color='red')  # line of perfect prediction
plt.savefig(f'{savedir}/{lv_data}_actual-predicted.png')
plt.show()


# # %% Plot fitted values from the model
#   # Extract fitted values for plotting
#   # df['fitted'] = result.fittedvalues

#   # Plot the results
#   # plt.plot(df['block'], df['fitted'], label=lv_data)
# plt.plot(result.fittedvalues)

# # Add labels and legend
# # plt.xlabel('Block')
# plt.ylabel('Fitted Values')
# plt.title(f'{lv_data} values by condition (extero/interoception)')
# # plt.legend(loc='best')

# # Save and show the plot
# plt.savefig(f'{savedir}/{lv_data}_fitted_vals.png')
# plt.show()

# %% Plot random effects to see how much variance is explained by each subject
# Extract random effects
random_effects = result.random_effects

# Convert to DataFrame
random_effects_df = pd.DataFrame(random_effects).T

# Plot the random effects
random_effects_df.plot(kind='bar', figsize=(10, 6))
plt.title('Random Effects (variance explained by each subject)')
plt.xlabel('Subject')
plt.ylabel('Random Effect Value')
plt.savefig(f'{savedir}/{lv_data}_subject_rand_effects.png')
plt.show()



# %% Fixed effects: condition and block

# # Plot fixed effects: lv_data by condition
# sns.lmplot(x='condition', y=lv_data, data=df, aspect=1.5, ci=None)
# plt.title(f'Fixed Effect of Condition on {lv_data} (Condition: extero=1, intero=0)')
# # plt.savefig(f'{savedir}/{lv_data}_fixed_effect_cond.png')
# plt.show()

# Plot fixed effects: lv_data by block
sns.lmplot(x='block', y=lv_data, data=df, aspect=1.5, ci=None)
plt.title(f'Effect of Block on LV1 Brainscore')
plt.savefig(f'{savedir}/{lv_data}_effect_of_block.png')
plt.show()




# %% Plot trajectories of subjects' cluster/brain scores over blocks with average line

# Define the conditions and blocks
conditions = df['condition'].unique()
blocks = df['block'].unique()

# Define the color palette
palette = sns.color_palette("hls", n_colors=len(df['subjID'].unique()))
colors = dict(zip(df['subjID'].unique(), palette))

# Functions --------------------------------------------------------------------------

# Get dynamic y-axis limits based on min and max subject scores
def get_y_axis_limits(df, lv_data):
    min_val = df[lv_data].min()
    max_val = df[lv_data].max()
    return min_val - 5, max_val + 5  # Add a margin to the min/max for better visibility

# Plotting function with upper and lower CI bounds and consistent y-axis limits
def plot_with_CI(df, ax, title, lv_data, colors, y_limits):
    # Plot each subject's trajectory
    for subj in df['subjID'].unique():
        subj_df = df[df['subjID'] == subj]
        ax.plot(subj_df['block'], subj_df[lv_data], marker='o', label=subj, color=colors[subj], alpha=0.6)
        ax.text(subj_df['block'].iloc[-1], subj_df[lv_data].iloc[-1], subj, 
                fontsize=9, color=colors[subj], ha='left', va='center')

    # Plot the mean and confidence intervals across subjects for each block
    mean_df = df.groupby('block', as_index=False)[lv_data].mean()
    ci_up = df.groupby('block')[f'{lv_data}_ci_up'].first().values
    ci_ll = df.groupby('block')[f'{lv_data}_ci_ll'].first().values

    ax.plot(mean_df['block'], mean_df[lv_data], marker='o', color='black', linewidth=2, linestyle='--', label='Mean')

    # Apply the consistent y-axis limits (based on overall max and min subject scores)
    ax.set_ylim(y_limits)

    ax.set_title(title)
    ax.set_xlabel('Block')
    ax.set_ylabel(f'{lv_data_string}')

# Plotting function without precalculated CIs, calculates 95% CIs
def plot_no_CI(df, ax, title, lv_data, colors, y_limits):
    # Plot each subject's trajectory
    for subj in df['subjID'].unique():
        subj_df = df[df['subjID'] == subj]
        ax.plot(subj_df['block'], subj_df[lv_data], marker='o', label=subj, color=colors[subj], alpha=0.6)
        ax.text(subj_df['block'].iloc[-1], subj_df[lv_data].iloc[-1], subj, 
                fontsize=9, color=colors[subj], ha='left', va='center')

    # Calculate mean and 95% CI across each block
    mean_df = df.groupby('block', as_index=False)[lv_data].mean()
    sem_df = df.groupby('block', as_index=False)[lv_data].sem()  # Standard error of the mean

    # Calculate 95% confidence intervals (mean ± 1.96 * SEM)
    ci_up = mean_df[lv_data] + 1.96 * sem_df[lv_data]
    ci_ll = mean_df[lv_data] - 1.96 * sem_df[lv_data]

    # Plot the mean line (don't plot shaded confidence interval here - too messy)
    ax.plot(mean_df['block'], mean_df[lv_data], marker='o', color='black', linewidth=2, linestyle='--', label='Mean')
    # ax.fill_between(mean_df['block'], ci_ll, ci_up, color='gray', alpha=0.3, label='95% CI')  # Shaded region

    # Apply the consistent y-axis limits
    ax.set_ylim(y_limits)

    ax.set_title(title)
    ax.set_xlabel('Block')
    ax.set_ylabel(f'{lv_data_string}')

# Code body --------------------------------------------------------------------------

# Get y-axis limits based on the left panel (first condition)
df_first_condition = df[df['condition'] == conditions[0]]
y_limits = get_y_axis_limits(df_first_condition, lv_data)  # Max and min subject scores; used to set y-axes range for plot
y_limits = [-1.3, 1.4] # manually set y-axis range if necessary

# Initiate figure with 3 panels (2 for conditions, 1 for overall mean)
fig, axes = plt.subplots(1, len(conditions) + 1, figsize=(16, 8))

# Plot for each condition
for i, condition in enumerate(conditions):
    condstring = 'Interoception' if condition == 0 else 'Exteroception'
    ax = axes[i]
    if 'bs' in lv_data:
        plot_with_CI(df[df['condition'] == condition], ax, f"{lv_data_string}: {condstring} Change Over Time", lv_data, colors, y_limits)
    else:
        plot_no_CI(df[df['condition'] == condition], ax, f"{lv_data_string}: {condstring} Change Over Time", lv_data, colors, y_limits)

# ----------------------------------------------------------------------------------
# Plot the combined mean with calculated confidence intervals in the rightmost panel
# ----------------------------------------------------------------------------------

ax = axes[-1]
all_mean_df = df.groupby('block', as_index=False)[lv_data].mean()
all_sem_df = df.groupby('block', as_index=False)[lv_data].sem()  # Calculate SEM for all conditions

# Calculate 95% CI for combined conditions
ci_up_combined = all_mean_df[lv_data] + 1.96 * all_sem_df[lv_data]
ci_ll_combined = all_mean_df[lv_data] - 1.96 * all_sem_df[lv_data]

# Plot the mean across conditions in black
ax.plot(all_mean_df['block'], all_mean_df[lv_data], marker='o', color='black', linewidth=2, label='Combined Mean')

# Plot the 95% CI shaded region
ax.fill_between(all_mean_df['block'], ci_ll_combined, ci_up_combined, color='gray', alpha=0.3, label='95% CI')

# Apply consistent y-axis limits
ax.set_ylim(y_limits)
ax.set_title(f'{lv_data_string}: Mean across Conditions with CI')
ax.set_xlabel('Block')
ax.set_ylabel(f'{lv_data_string} mean across subjects')
ax.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig(f'{savedir}/{lv_data}_change_over_blocks_CI.png')
plt.show()


# %%
