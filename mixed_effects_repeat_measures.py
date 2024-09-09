# Mixed Effects Linear Model with Repeated Measures for LPS Brain Scores

# Updated 2024-09-04
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
import seaborn as sns
import statsmodels.formula.api as smf
import re


# Set paths
spat_dir = f'/home/common/piaf/LPS_STHLM/analysis_2023/PLS/int_rs_denoised_newconds/1_ROIS/all_rois_n8_NVoxels2357/rmoutliers_20conds_meanInteroExtero_pl_all_pls_1_n16' # dir of the matlab SPAT results
scores_csv = f'{spat_dir}/rmoutliers_20conds_meanInteroExtero_pl_all_pls_1_n16_extracted_mean_values.csv' # full path to brain scores/clusters/LVs CSV file
savedir = '/home/common/piaf/LPS_STHLM/analysis_2023/PLS/mixed_effects_models_brainscores/pl_intero-extero_allblocks_n16'

# spat_dir = f'/home/common/piaf/LPS_STHLM/analysis_2023/PLS/int_rs_denoised_newconds/1_ROIS/all_rois_n8_NVoxels2357/rmoutliers_20conds_meanInteroExtero_lps_all_pls_1_n22' # dir of the matlab SPAT results
# scores_csv = f'{spat_dir}/rmoutliers_20conds_meanInteroExtero_lps_all_pls_1_n22_extracted_mean_values.csv' # full path to brain scores/clusters/LVs CSV file
# savedir = '/home/common/piaf/LPS_STHLM/analysis_2023/PLS/mixed_effects_models_brainscores/lps_intero-extero_allblocks'

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
  match = re.match(r'mean_(\w+)_block(\d+)_pl', condition_block)
  # match = re.match(r'mean_(\w+)_block(\d+)_lps', condition_block)
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
lv_data = 'bs_lv1' # specify brain score or latent variable you want to model
lv_data_string = 'Brainscore LV1' # string for titles and plot headers and such

df = pd.concat([df_reshape['subjID'], df_reshape['condition'], df_reshape['block'], df_reshape[lv_data]], axis=1)

# Model cluster values as a function of 'condition' with random effects for 'subject' and 'block'
model = smf.mixedlm(f"{lv_data} ~ block", df, groups=df["subjID"]) # acknowledges that effect of block on brainscores ~ conditions may differ between subjects
# model = smf.mixedlm(f"{lv_data} ~ condition", df, groups=df["subjID"], re_formula="~block") # try with groups=df["block"] (averages across subjects?)
result = model.fit()

# Print the summary of the model
print(result.summary())


# %% 1-way ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

lv_data = 'bs_lv3' # specify brain score or latent variable you want to model

# Don't need condition here (only have one)
df = pd.concat([df_reshape['subjID'], df_reshape['block'], df_reshape[lv_data]], axis=1)

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
# Recreate above 3-panel plot, but show the averaged line across subjects
# lv_data_string = 'LV1 Cluster1 (threshold +/-2)'  # Choose a title for the brain score/cluster for plots
conditions = df['condition'].unique()  # Define conditions

y_error = np.array([2.1767, 2.8459]) # get from SPAT report BSR range

# # drop one subject row
# i = df[(df.subjID == 'sub-105')].index
# df = df.drop(i)

# fig, axes = plt.subplots(1, len(conditions), figsize=(6, 4))  # Initiate figure with one panel
fig, axes = plt.subplots(1, len(conditions) + 1, figsize=(24, 8))  # Initiate figure with 3 panels


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
    print(mean_df)
    ax.plot(mean_df['block'], mean_df[lv_data], marker='o', color='black', linewidth=2, linestyle='--', label='Mean')

    ax.set_title(title)
    ax.set_xlabel('Block')
    ax.set_ylabel(f'{lv_data_string}') # maybe don't need y-axis label (model fitted values, i.e. y_hat based on intero/extero condition )
    ax.set_ylim([-65,45]) # set all y-axis limits to this range
    # ax.legend(loc='best')

# Plot each condition separately
for i, condition in enumerate(conditions):
    
    # Get labels for conditions
    if condition == 0:
        condstring = 'Interoception'
    elif condition == 1:
        condstring = 'Exteroception'

    ax = axes[i]
    plot_with_labels_avg(df[df['condition'] == condition], ax, f"{lv_data_string}: {condstring} Change Over Time", lv_data, colors)

# # Plot one condition only
# condition = 1
# condstring = 'Interoception'
# ax = axes[1]
# plot_with_labels_avg(df[df['condition'] == condition], axes, f"{lv_data_string}: {condstring} Change Over Time", lv_data, colors)

# Plot only mean lines on last panel
mean_colors = ['red', 'blue']
ax = axes[-1]
# for i, condition in enumerate(conditions):
for condition in conditions:

    # Get labels for conditions
    if condition == 0:
        condstring = 'Interoception'
    elif condition == 1:
        condstring = 'Exteroception'

    df_cond = df[df['condition'] == condition]
    mean_dfc = df_cond.groupby('block', as_index=False)[lv_data].mean()
    ax.plot(mean_dfc['block'], mean_dfc[lv_data], marker='o', color=mean_colors[condition], linewidth=2, label=condstring)
    # ax.plot(mean_df['block'], mean_df[lv_data], marker='o', color='black', linewidth=2, linestyle='--', label='Mean')

    # add error bar
    y_err = np.multiply(np.ones([2,len(mean_dfc['block'])]).transpose(), y_error).transpose()
    ax.errorbar(mean_dfc['block'], mean_dfc[lv_data],yerr=y_err, color=mean_colors[condition]) # error bar markers
    ax.fill_between(mean_dfc['block'], mean_dfc[lv_data]-y_err[0,], mean_dfc[lv_data]+y_err[1,], color=mean_colors[condition], alpha=0.4) # shaded error bars

ax.set_title(f'Intero- vs. Exteroception Average Effects on {lv_data_string}')
ax.set_xlabel('Block')
ax.set_ylabel(f'{lv_data_string} average across subjects')
ax.set_ylim([-65,45]) # set all y-axis limits to this range
ax.legend()


# # Plot the averaged trajectories in the last subplot
# # df_avg = df.groupby(['subjID', 'block'], as_index=False)[lv_data].mean()
# ax = axes[-1]
# # plot_with_labels_avg(df_avg, ax, f'Intero- and Exteroception Averaged Effect on {lv_data_string}', lv_data, colors)
# # Calculate and plot the mean across subjects for each block
# mean_df = df.groupby('block', as_index=False)[lv_data].mean()
# ax.plot(mean_df['block'], mean_df[lv_data], marker='o', color='black', linewidth=2, linestyle='--', label='Mean')



# Adjust the layout and show the plot
plt.tight_layout()
# plt.legend(loc='best', ncol=2, bbox_to_anchor=(1, 1)) # define legend (lists subjects)
plt.savefig(f'{savedir}/{lv_data}_change_over_blocks_avg_err.png') # save figure as PNG
plt.show()


# %% Plot ANOVA results
# Calculate means and standard errors for each block
means = df.groupby('block')[lv_data].mean()
errors = df.groupby('block')[lv_data].sem()

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(means.index, means, yerr=errors, capsize=5, color='skyblue')

# Add labels and title
plt.xlabel('Block')
plt.ylabel(f'{lv_data} Means and Standard Deviation')
plt.title(f'Effect of Block on {lv_data}')
plt.savefig(f'{savedir}/{lv_data}_effect_of_block_anova.png') # save figure as PNG
plt.show()

# %%
