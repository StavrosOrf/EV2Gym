
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


file_path = './results/eval_10cs_1tr_V2GProfitPlusLoads_13_algos_2_exp_2024_07_03_919059/data.csv'
file_path = './results/eval_10cs_1tr_V2GProfitPlusLoads_12_algos_2_exp_2024_07_03_304766/data.csv'

# good one for tr power limit
file_path = './results/eval_10cs_1tr_V2GProfitPlusLoads_14_algos_5_exp_2024_07_03_848609/data.csv'

data = pd.read_csv(file_path)

print(data.columns)

#print unique algorithms
print(data['Algorithm'].unique())

columns_to_keep = ['run',
                   'Algorithm',
                   'power_limit',
                   'total_profits',
                   'total_transformer_overload',
                   'average_user_satisfaction']

# keep only alagorithm == TQC
# data = data[data['Algorithm'] == 'TQC']

algorithms_to_keep = [
    'ChargeAsFastAsPossible',
    # 'ChargeAsLateAsPossible',
    # PPO, A2C,
    # DDPG,
    # SAC,
    # TD3,
    'SAC',
    'TQC',
    # 'TRPO',
    # ARS,
    # RecurrentPPO,
    # SAC,
    # TQC,
    # # TD3,
    # # ARS,
    # # RecurrentPPO,
    # RoundRobin,
    'eMPC_V2G',
    # # V2GProfitMaxLoadsOracle,
    'V2GProfitMaxOracleGB',
    # V2GProfitMaxOracle,
    # PowerTrackingErrorrMin
]

data = data[data['Algorithm'].isin(algorithms_to_keep)]

#change algorithm names
data['Algorithm'] = data['Algorithm'].replace({
    'ChargeAsFastAsPossible': 'AFAP',
    'eMPC_V2G': 'MPC',
    'V2GProfitMaxOracleGB': 'Optimal',
})
    

print(data[columns_to_keep].head(25))


# plot the results

sns.set_theme(style="whitegrid")

# plot total profits, total transformer overload, and energy user satisfaction in three parallel subplots
# grouping runs by algorithm and power limit

# fig, axs = plt.subplots(1, 3, figsize=(10, 5))

# sns.lineplot(data=data,
#              x='power_limit',
#              y='total_profits',
#              hue='Algorithm',
#              marker='o',
#              markersize=10,
#              #  markerfacecolor='black',
#              err_style='bars',
#             #  errorbar="sd",
#              err_kws={'capsize': 5},
#              ax=axs[0])

labely_fontsize = 16
labelx_fontsize = 15

fig = plt.figure()
#set font size
plt.rcParams.update({'font.size': 22})
# set font style
plt.rcParams.update({'font.family': 'serif'})

ax1 = fig.add_subplot(131)

sns.barplot(data=data,
            x='power_limit',
            y='total_profits',
            hue='Algorithm',
            capsize=0.4,
            # kind="bar",
            ax=ax1)

ax1.set_ylabel('Total Profits (€)', fontsize=labely_fontsize)
ax1.set_xlabel('Power Limit (kW)', fontsize=labelx_fontsize)
ax1.get_legend().remove()

ax2 = fig.add_subplot(132)
sns.barplot(data=data,
            x='power_limit',
            y='total_transformer_overload',
            hue='Algorithm',
            capsize=0.4,
            ax=ax2)
ax2.set_ylabel('Total Transformer Overload (kW)', fontsize=labely_fontsize)
ax2.set_xlabel('Power Limit (kW)', fontsize=labelx_fontsize)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=6, prop={'size': 15})

ax3 = fig.add_subplot(133)
sns.barplot(data=data,
            x='power_limit',
            y='average_user_satisfaction',
            hue='Algorithm',
            capsize=0.4,
            ax=ax3)

ax3.set_ylabel('User Satisfaction (%)', fontsize=labely_fontsize)
ax3.set_xlabel('Power Limit (kW)', fontsize=labelx_fontsize)
ax3.get_legend().remove()
# ax3.set_xticks(data['power_limit'].unique())


# y-axis labels
# axs[0].set_ylabel('Total Profits (€)')
# axs[0].set_xlabel('Power Limit (kW)')
# axs[0].set_xticks(data['power_limit'].unique())

# plt.subplot(1, 3, 2)

# set_labels = ['Total Profits (€)', 'Total Transformer Overload (kW)', 'User Satisfaction (%)']

# g = sns.catplot(data=data,
#             x='power_limit',
#             y='total_transformer_overload',
#             hue='Algorithm',
#             # kind='point',
#             # markers='o',
#             # markersize=10,
#             kind="bar",
#             #  markerfacecolor='black',
#             # err_style='bars',
#             # #  errorbar="sd",
#             # err_kws={'capsize': 5},
# )

# y-axis labels


# sns.lineplot(data=data,
#              x='power_limit',
#              y='total_transformer_overload',
#              hue='Algorithm',
#              marker='o',
#              markersize=10,
#              #  markerfacecolor='black',
#              err_style='bars',
#             #  errorbar="sd",
#              err_kws={'capsize': 5},
#              ax=axs[1])

# y-axis labels
# axs[1].set_ylabel('Total Transformer Overload (kW)')
# axs[1].set_xlabel('Power Limit (kW)')
# axs[1].set_xticks(data['power_limit'].unique())

# sns.lineplot(data=data,
#              x='power_limit',
#              y='average_user_satisfaction',
#              hue='Algorithm',
#             #  errorbar="sd",
#              err_kws={'capsize': 5},
#              marker='o',
#              markersize=10,
#              #  markerfacecolor='black',
#              err_style='bars',
#              ax=axs[2])

# y-axis labels
# axs[2].set_ylabel('User Satisfaction (%)')
# axs[2].set_xlabel('Power Limit (kW)')
# axs[2].set_xticks(data['power_limit'].unique())


# # put legend under the plots once
# axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)

# dissapear the legends of ax[0] and ax[2]
# axs[0].get_legend().remove()
# axs[2].get_legend().remove()

plt.show()
