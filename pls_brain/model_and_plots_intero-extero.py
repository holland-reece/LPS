# Mixed Effects Linear Model with Repeated Measures: Intero- vs. Exteroception Mean-Centered PLS Brain Scores

# Updated 2024-09-19
# Created 2024-08-06

# DESCRIPTION
    # models interoception ~ exteroception cluster scores (and/or brain scores) with repeated measures (blocks)
    # plots subjects' brainscores (for one LV) over time, then the averages per condition in a 3-panel figure (requires you to input error margins from PLS reports)

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
spat_dir = f'/home/common/piaf/LPS_STHLM/analysis_2023/PLS/int_rs_denoised_newconds/1_ROIS/all_rois_n8_NVoxels2357/rmoutliers_20conds_meanInteroExtero_pl_all_pls_1_n16' # dir of the matlab SPAT results
scores_csv = f'{spat_dir}/rmoutliers_20conds_meanInteroExtero_pl_all_pls_1_n16_extracted_mean_values.csv' # full path to brain scores/clusters/LVs CSV file
savedir = '/home/common/piaf/LPS_STHLM/analysis_2023/PLS/mixed_effects_models_brainscores/pl_intero-extero_allblocks_n16'

# Create results directory
if os.path.isdir(savedir)==False:
  cmd = f'mkdir {savedir}'
  subprocess.run(cmd, shell=True, executable='/bin/bash')

# Load data
df_init = pd.read_csv(scores_csv, sep=',', header=0)
df_init.set_index('filename', inplace=True)

# # Define conditions as integers
# interoception = 0
# exteroception = 1

# %% Reshape the df_init to include only brain scores or clusters; leave out paths and groups (groups column is all ones)

# df_reshape = pd.melt(df.reset_index(), id_vars=['filename'], var_name='condition_block', value_name='value') # value can either be cluster, brain score, or LV
cluster_cols = [col for col in df_init.columns if 'thp2n2_cluster' in col] # list cluster column names
brainscore_cols = [col for col in df_init.columns if 'bs_' in col] # list brain score column names
df_reshape = pd.concat([df_init['subjID'], df_init['condition'], df_init[cluster_cols], df_init[brainscore_cols]], axis=1).reset_index(drop=True) # create new df with only paths, conditions, and cluster values
df_reshape.rename(columns={'condition': 'condition_block'}, inplace=True) # rename column headers

# Extract condition and block using regex
def parse_condition_block(condition_block):
  match = re.match(r'mean_(\w+)_block(\d+)_pl', condition_block)
#   match = re.match(r'mean_(\w+)_block(\d+)_lps', condition_block)
  if match:
      condition = match.group(1)
      block = int(match.group(2))
    #   print(f"{condition}, {block}") # TEST
      return condition, block
  else:
    return None, None
    
# Separate condition and block into their own columns
df_reshape[['condition', 'block']] = df_reshape['condition_block'].apply(lambda x: pd.Series(parse_condition_block(x))) # apply function to separate cond and block
df_reshape = df_reshape.drop('condition_block', axis=1) # now condition and block have their own columns so we can remove this one
df_reshape = df_reshape[['subjID', 'condition', 'block'] + [ c for c in df_reshape if c not in ['subjID', 'condition', 'block']]] # move condition and block cols to left

# Replace condition names (pandas objects) with integers 0 and 1
df_reshape.replace('interoception', 0, inplace=True)
df_reshape.replace('exteroception', 1, inplace=True)

# %% Define and fit the model
# lv_data = 'bs_lv3' # specify brain score or latent variable you want to model
# lv_data_string = 'Brainscore LV3' # string for titles and plot headers and such
lv_data = 'lv1_thp2n2_cluster2'
lv_data_string = 'LV1 Cluster 2: Right Mid Cingulate (thr +/-2)'

df = pd.concat([df_reshape['subjID'], df_reshape['condition'], df_reshape['block'], df_reshape[lv_data]], axis=1)

# Model cluster values as a function of 'condition' with random effects for 'subject' and 'block'
model = smf.mixedlm(f"{lv_data} ~ block", df, groups=df["subjID"]) # acknowledges that effect of block on brainscores ~ conditions may differ between subjects
# model = smf.mixedlm(f"{lv_data} ~ condition", df, groups=df["subjID"], re_formula="~block") # try with groups=df["block"] (averages across subjects?)
result = model.fit()

# Print the summary of the model
print(result.summary())


# %% 1-way ANOVA

# Don't need condition here (only have one)
df = pd.concat([df['subjID'], df_reshape['block'], df_reshape[lv_data]], axis=1)

# If using repeated measures ANOVA (AnovaRM) with subject as a random effect:
anova_rm = AnovaRM(df, depvar=lv_data, subject='subjID', within=['block'])
result = anova_rm.fit()

# Print the summary of the repeated measures ANOVA
print(result.summary())


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


# %% Plot trajectories of subjects' cluster/brain scores over blocks (by condition, then averaged)
# import seaborn as sns

lv_data_string = 'LV1 Cluster1 (thresh +/-2) value' # Choose a title for the brain score/cluster for plots
conditions = df['condition'].unique() # Define conditions

fig, axes = plt.subplots(1, len(conditions) + 1, figsize=(24, 8)) # Initiate figure

# Define the color palette
palette = sns.color_palette("hls", n_colors=len(df['subjID'].unique()))
colors = dict(zip(df['subjID'].unique(), palette))

# Plotting function with labels directly on lines
def plot_with_labels(df, ax, title, lv_data, colors):
    for subj in df['subjID'].unique():
        subj_df = df[df['subjID'] == subj]
        ax.plot(subj_df['block'], subj_df[lv_data], marker='o', label=subj, color=colors[subj])

        # Label the line at the last data point
        ax.text(subj_df['block'].iloc[-1], subj_df[lv_data].iloc[-1], subj, 
                fontsize=9, color=colors[subj], ha='left', va='center')
    
    ax.set_title(title)
    ax.set_xlabel('Block')
    ax.set_ylabel(lv_data)


# Plot each condition separately

for i, condition in enumerate(conditions):
    
    # Get labels for conditions
    if condition == 0:
       condstring = 'Interoception'
    elif condition == 1:
       condstring = 'Exteroception'

    ax = axes[i]
    plot_with_labels(df[df['condition'] == condition], ax, f"{lv_data_string}: {condstring}", lv_data, colors)

# Calculate the mean of lv_data for each subject across conditions
df_avg = df.groupby(['subjID', 'block'], as_index=False)[lv_data].mean()
ax = axes[-1]
plot_with_labels(df_avg, ax, f'Averaged {lv_data_string} Across Conditions', lv_data, colors)

# Adjust the layout and show the plot
plt.tight_layout()
plt.legend(loc='best', ncol=2, bbox_to_anchor=(1, 1))
# plt.savefig(f'{savedir}/{lv_data}_change_over_blocks.png')
plt.show()


# %% Plot trajectories of subjects' cluster/brain scores over blocks with average line

# remove one subject (if range is causing plotting error)
df.drop(df[df['subjID']=='sub-105'].index, inplace=True)

# manually set y-axis range if necessary
y_limits = [-1.1, 1.2]

conditions = df['condition'].unique()  # Define conditions

fig, axes = plt.subplots(1, len(conditions) + 1, figsize=(24, 8))  # Initiate figure with 3 panels
# fig, axes = plt.subplots(1, len(conditions) + 1, figsize=(3, 1))  # Adjust the figure size to avoid error

# Define the color palette
palette = sns.color_palette("hls", n_colors=len(df['subjID'].unique()))
colors = dict(zip(df['subjID'].unique(), palette))

# Plotting function with labels directly on lines and average line
def plot_with_labels_avg(df, ax, title, lv_data, colors):
    for subj in df['subjID'].unique():
        subj_df = df[df['subjID'] == subj]
        ax.plot(subj_df['block'], subj_df[lv_data], marker='o', label=subj, color=colors[subj], alpha=0.6)

        # Label the line at the last data point
        ax.text(subj_df['block'].iloc[-1], subj_df[lv_data].iloc[-1], subj, 
                fontsize=9, color=colors[subj], ha='left', va='center')

    # Calculate and plot the mean across subjects for each block
    mean_df = df.groupby('block', as_index=False)[lv_data].mean()
    ax.plot(mean_df['block'], mean_df[lv_data], marker='o', color='black', linewidth=2, linestyle='--', label='Mean')

    ax.set_title(title)
    ax.set_xlabel('Block')
    ax.set_ylabel(f'{lv_data_string}')
    ax.set_ylim(y_limits)  # Set all y-axis limits to this range

# Plot each condition separately
for i, condition in enumerate(conditions):
    
    # Get labels for conditions
    if condition == 0:
        condstring = 'Interoception'
    elif condition == 1:
        condstring = 'Exteroception'

    ax = axes[i]
    plot_with_labels_avg(df[df['condition'] == condition], ax, f"{lv_data_string}: {condstring}", lv_data, colors)

# Plot only the overall mean with 95% CI on the last panel
ax = axes[-1]

# Calculate the mean and standard error of the mean (SEM) across all conditions
mean_df = df.groupby('block', as_index=False)[lv_data].mean()
sem_df = df.groupby('block', as_index=False)[lv_data].sem()

# Calculate the 95% CI: mean ± 1.96 * SEM (for a 95% confidence interval)
ci_up = mean_df[lv_data] + 1.96 * sem_df[lv_data]
ci_low = mean_df[lv_data] - 1.96 * sem_df[lv_data]

# Plot the overall mean line
ax.plot(mean_df['block'], mean_df[lv_data], marker='o', color='black', linewidth=2, label='Mean')

# Plot the 95% CI as a shaded region
ax.fill_between(mean_df['block'], ci_low, ci_up, color='gray', alpha=0.4, label='95% CI')

ax.set_title(f'Mean {lv_data_string} with 95% CI')
ax.set_xlabel('Block')
ax.set_ylabel(f'{lv_data_string} mean across subjects')
ax.set_ylim(y_limits)  # Set all y-axis limits to this range
ax.legend()

# Adjust the layout and show the plot
# plt.tight_layout()
plt.savefig(f'{savedir}/{lv_data}_change_over_blocks_avg_err.png')  # Save figure as PNG
plt.show()


# %%
