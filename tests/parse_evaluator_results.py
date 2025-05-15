
import pandas as pd


data = pd.read_csv('./results/eval_10cs_1tr_V2G_MPC_4_algos_2_exp_2025_05_15_243325/data.csv')


print(data.shape)

# group by algotithm and get mean and std
columns = ['Unnamed: 0', 'run', 'Algorithm', 'total_ev_served', 'total_profits',
           'total_energy_charged', 'total_energy_discharged',
           'average_user_satisfaction', 'power_tracker_violation',
           'tracking_error', 'energy_tracking_error', 'energy_user_satisfaction',
           'total_transformer_overload', 'battery_degradation',
           'battery_degradation_calendar', 'battery_degradation_cycling',
           'total_reward']

# columns_to_keep = ['Algorithm',
#                    'run',                   
#                    'total_profits',
#                    'voltage_violation',
#                    'voltage_violation_counter',
#                    'average_user_satisfaction',                                      
#                    'total_energy_charged',
#                    'total_energy_discharged',
#                    'total_reward',
#                    'time',
#                    ]

columns_to_keep = ['Algorithm',
                   'run',
                   'total_profits',
       'total_energy_charged',
       'total_energy_discharged',
       'average_user_satisfaction',
       'total_transformer_overload',
       'battery_degradation',
       'battery_degradation_calendar',
       'battery_degradation_cycling',
       'total_reward',
       'time']

data = data[columns_to_keep]
print(data.head(20))

columns_to_drop = [
    'run',
]

data = data.drop(columns=columns_to_drop)


data_grouped = data.groupby('Algorithm').agg(['mean', 'std'])

# create new columns with the mean and std of the total_energy_charged combined as a string
data_grouped['total_energy_charged'] = data_grouped['total_energy_charged']\
    .apply(lambda x: f"${x['mean']/1000:.1f}$ ±${x['std']/1000:.1f}$", axis=1)
data_grouped['total_energy_discharged'] = data_grouped['total_energy_discharged']\
    .apply(lambda x: f"${x['mean']/1000:.2f}$ ±${x['std']/1000:.2f}$", axis=1)
data_grouped['average_user_satisfaction'] = data_grouped['average_user_satisfaction']\
    .apply(lambda x: f"${x['mean']*100:.1f}$ ±${x['std']*100:.1f}$", axis=1)
data_grouped['total_profits'] = data_grouped['total_profits']\
    .apply(lambda x: f"${x['mean']:.0f}$ ±${x['std']:.0f}$", axis=1)
data_grouped['total_transformer_overload'] = data_grouped['total_transformer_overload']\
    .apply(lambda x: f"${x['mean']/1000:.3f}$ ±${x['std']/1000:.3f}$", axis=1)
data_grouped['battery_degradation'] = data_grouped['battery_degradation']\
    .apply(lambda x: f"${x['mean']*10000:.2f}$ ±${x['std']*10000:.2f}$", axis=1)
data_grouped['battery_degradation_calendar'] = data_grouped['battery_degradation_calendar']\
       .apply(lambda x: f"${x['mean']*10000:.2f}$ ±${x['std']*10000:.2f}$", axis=1)
data_grouped['battery_degradation_cycling'] = data_grouped['battery_degradation_cycling']\
       .apply(lambda x: f"${x['mean']*10000:.2f}$ ±${x['std']*10000:.2f}$", axis=1)
       
data_grouped['total_reward'] = data_grouped['total_reward']\
    .apply(lambda x: f"${x['mean']:.3f}$ ±${x['std']:.3f}$", axis=1)
data_grouped['time'] = data_grouped['time']\
    .apply(lambda x: f"${x['mean']/300:.3f}$", axis=1)

# rearange rows


# drop the mean and std columns
data_grouped = data_grouped.droplevel(1, axis=1)
# print the results
# drop duplicate columns
data_grouped = data_grouped.loc[:, ~data_grouped.columns.duplicated()]
# rename columns
data_grouped.columns = ['Costs [€]',                        
                        'Energy Ch. [MWh]',
                        'Energy Dis. [MWh]',
                        'User Satisfaction [\%]',
                            'Transformer Overload [kW]',
                            'B Degrad [\%]',
                            'B Degrad Cal [\%]',
                            'B Degrad Cyc [\%]',
                        'Reward [-]',
                        'Step time [sec/step]',
                        ]

       #             'total_profits',
       # 'total_energy_charged',
       # 'total_energy_discharged',
       # 'average_user_satisfaction',
       # 'total_transformer_overload',
       # 'battery_degradation',
       # 'battery_degradation_calendar',
       # 'battery_degradation_cycling',
       # 'total_reward',
       # 'time']

print(data_grouped)

# rename algorithm names with shorter names
data_grouped.index = data_grouped.index.str.replace(
    'ChargeAsFastAsPossibleToDesiredCapacity', 'AFAP')
# data_grouped.index = data_grouped.index.str.replace(
#     'QT', 'Q-DT')
# data_grouped.index = data_grouped.index.str.replace('eMPC_G2V', 'MB-TD3')
# data_grouped.index = data_grouped.index.str.replace(
#     'RandomAgent', 'Random Charging')
# data_grouped.index = data_grouped.index.str.replace('GNN_act_emb_DT', 'GNN-DT')


# change order of rows
data_grouped = data_grouped.reindex(['AFAP',
                                     'PPO',
                                     'eMPC_G2V',
                                     'eMPC_V2G',
                                     ])


# rename PowerTrackingErrorrMin to Optimal
# print(data_grouped)
print(data_grouped.to_latex())


exit()
# % \usepackage{tabularray}
# \begin{table}
# \centering
# \captionsetup{labelformat=empty}
# \caption{Analysis of the Overall Reward into its Three Components}
# \label{tab:reward_breakdown}
# \begin{tblr}{
#   cells = {c,t},
#   vline{2} = {2-7}{0.05em},
#   hline{1,8} = {-}{0.08em},
#   hline{2} = {-}{0.05em},
# }
# Algorithm & {Energy Charged\\~[MWh]} & {Energy Discharged\\~[MWh]} & {User Satisfaction\\~[\%]} & {Power Violation\\~[kW]} & Costs~[€] & Reward [10-5] & {Exec. Time\\~[sec/step]}\\
# CAFAP & $1.3$ ±$0.2$ & $0.00$ ±$0.00$ & $100.0$ ±$0.0$ & $1289.2$ ±$261.8$ & $-277$ ±$165$ & $-1.974$ ±$0.283$ & $0.001$\\
# BaU & $1.3$ ±$0.2$ & $0.00$ ±$0.00$ & $99.9$ ±$0.2$ & $10.5$ ±$9.4$ & $-255$ ±$156$ & $-0.679$ ±$0.067$ & $0.001$\\
# DT & $0.9$ ±$0.1$ & $0.03$ ±$0.01$ & $94.4$ ±$1.6$ & $58.7$ ±$28.3$ & $-173$ ±$104$ & $-0.462$ ±$0.093$ & $0.006$\\
# Q-DT & $1.0$ ±$0.1$ & $0.00$ ±$0.00$ & $93.6$ ±$2.1$ & $20.1$ ±$21.4$ & $-187$ ±$113$ & $-0.665$ ±$0.135$ & $0.010$\\
# GNN-DT & $0.9$ ±$0.1$ & $0.19$ ±$0.03$ & $99.3$ ±$0.2$ & $21.7$ ±$22.8$ & $-142$ ±$89$ & $-0.027$ ±$0.023$ & $0.023$\\
# Optimal (Offline) & $1.9$ ±$0.2$ & $1.08$ ±$0.19$ & $99.1$ ±$0.2$ & $2.0$ ±$4.6$ & $-119$ ±$84$ & $-0.020$ ±$0.015$ & -
# \end{tblr}
# \end{table}


# ############################

# OLD CODE

# ############################# ############################# ############################
# ############################
# ############################
# ############################
# ############################
# ############################

#%%
import pandas as pd
#%%

# data = pd.read_csv('./results/data_MPC.csv')
# data = pd.read_csv('../results/eval_10cs_1tr_V2GProfitPlusLoads_13_algos_20_exp_2024_03_10_797555/data.csv')
# data = pd.read_csv('../results/eval_30cs_1tr_V2G_MPC2_5_algos_1_exp_2024_03_14_305094/data.csv')
data = pd.read_csv('./results/eval_10cs_1tr_V2G_MPC_4_algos_2_exp_2025_05_15_243325/data.csv')
#keep only rows where run is 0
# data = data[data['run'] == 3]

columns_to_keep = ['Algorithm','total_profits',
       'total_energy_charged', 'total_energy_discharged',
       'average_user_satisfaction',
       'total_transformer_overload',
       'battery_degradation',
       'battery_degradation_calendar', 'battery_degradation_cycling',
       # 'total_reward',
       'time']

data = data[columns_to_keep]

# create new columns with rounded values
data['total_profits'] = data['total_profits'].apply(lambda x: f"${x:.1f}$")
data['total_energy_charged'] = data['total_energy_charged'].apply(lambda x: f"${x:.1f}$")
data['total_energy_discharged'] = data['total_energy_discharged'].apply(lambda x: f"${x:.1f}$")
data['average_user_satisfaction'] = data['average_user_satisfaction'].apply(lambda x: f"${x*100:.0f}$")
data['total_transformer_overload'] = data['total_transformer_overload'].apply(lambda x: f"${x:.1f}$")
data['battery_degradation'] = data['battery_degradation'].apply(lambda x: f"${x*1000:.2f}$")
data['time'] = data['time'].apply(lambda x: f"${x:.2f}$")

print(data.columns)

data.columns = ['Algorithm',
       'Profits', 'Energy Charged',
                        'Energy Discharged', 'User Satisfaction',
                        'Transformer Overload',
                        'Battery Degradation',
                        'Battery Degradation Calendar', 'Battery Degradation Cycling',
                        'Time']

print(data)
print(data.to_latex())
exit()
#%%
# Plot total capacity lost (battery degradation) for each algorithm as a histogram for each run
data = pd.read_csv('C:/Users/stayr/Downloads/data_1.csv')
print(data.columns)

columns_to_keep = ['run','Algorithm','battery_degradation']

#use seaborn to plot the histogram
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

data = data[columns_to_keep]

#plot the histogram
# g = sns.FacetGrid(data, col="Algorithm",row='run')
sns.histplot("Algorithm", "battery_degradation", palette="viridis")
g.set_xticklabels(rotation=90)
plt.show()


#%%
# data = pd.read_csv('data_V2GProfitPlusLoads.csv')
data = pd.read_csv('../results/eval_10cs_1tr_V2GProfitPlusLoads_13_algos_20_exp_2024_03_10_797555/data.csv')
data = pd.read_csv('C:/Users/stayr/Downloads/data_1.csv')

columns_to_keep = ['Algorithm','total_profits',
       'total_energy_charged', 'total_energy_discharged',
       'average_user_satisfaction',
       'total_transformer_overload',
       'battery_degradation',
       'battery_degradation_calendar', 'battery_degradation_cycling',
       'total_reward',
       'time']

data = data[columns_to_keep]
data_grouped = data.groupby('Algorithm').agg(['mean', 'std'])

# create new columns with the mean and std of the total_energy_charged combined as a string
data_grouped['total_profits'] = data_grouped['total_profits']\
       .apply(lambda x: f"${x['mean']:.1f}$ ±${x['std']:.1f}$", axis=1)
data_grouped['total_energy_charged'] = data_grouped['total_energy_charged']\
       .apply(lambda x: f"${x['mean']:.0f}$ ±${x['std']:.0f}$", axis=1)
data_grouped['total_energy_discharged'] = data_grouped['total_energy_discharged']\
       .apply(lambda x: f"${x['mean']:.0f}$ ±${x['std']:.0f}$", axis=1)
data_grouped['average_user_satisfaction'] = data_grouped['average_user_satisfaction']\
       .apply(lambda x: f"${x['mean']*100:.0f}$ ±${x['std']*100:.0f}$", axis=1)
data_grouped['total_transformer_overload'] = data_grouped['total_transformer_overload']\
       .apply(lambda x: f"${x['mean']:.0f}$ ±${x['std']:.0f}$", axis=1)
data_grouped['battery_degradation'] = data_grouped['battery_degradation']\
       .apply(lambda x: f"${x['mean']*10000:.2f}$ ±${x['std']*10000:.2f}$", axis=1)
data_grouped['battery_degradation_calendar'] = data_grouped['battery_degradation_calendar']\
       .apply(lambda x: f"${x['mean']*10000:.2f}$ ±${x['std']*10000:.2f}$", axis=1)
data_grouped['battery_degradation_cycling'] = data_grouped['battery_degradation_cycling']\
       .apply(lambda x: f"${x['mean']*10000:.2f}$ ±${x['std']*10000:.2f}$", axis=1)
data_grouped['total_reward'] = data_grouped['total_reward']\
       .apply(lambda x: f"${x['mean']/1000:.1f}$ ±${x['std']/1000:.1f}$", axis=1)
data_grouped['time'] = data_grouped['time']\
       .apply(lambda x: f"${x['mean']:.2f}$ ±${x['std']:.2f}$", axis=1)
       
# drop the mean and std columns
data_grouped = data_grouped.droplevel(1, axis=1)
# drop duplicate columns
data_grouped = data_grouped.loc[:,~data_grouped.columns.duplicated()]

#rename columns

data_grouped.columns = ['Profits/Costs', 'Energy Charged',
                            'Energy Discharged', 'User Satisfaction',
                            'Transformer Overload',
                            'Battery Degradation', 'Battery Degradation Calendar',
                            'Battery Degradation Cycling', 'Reward', 'Time']

print(data_grouped)
print(data_grouped.to_latex())

#%%
data = pd.read_csv('data_Profitmax.csv')

columns_to_keep = ['Algorithm','total_profits',
       'total_energy_charged', 'total_energy_discharged',
       'average_user_satisfaction',
       'total_transformer_overload',
       'total_reward']
                   
data = data[columns_to_keep]

data_grouped = data.groupby('Algorithm').agg(['mean', 'std'])
# create new columns with the mean and std of the total_energy_charged combined as a string 
data_grouped['total_profits'] = data_grouped['total_profits']\
       .apply(lambda x: f"${x['mean']:.1f}$ ±${x['std']:.1f}$", axis=1)
data_grouped['total_energy_charged'] = data_grouped['total_energy_charged']\
       .apply(lambda x: f"${x['mean']:.0f}$ ±${x['std']:.0f}$", axis=1)
data_grouped['total_energy_discharged'] = data_grouped['total_energy_discharged']\
       .apply(lambda x: f"${x['mean']:.0f}$ ±${x['std']:.0f}$", axis=1)
data_grouped['average_user_satisfaction'] = data_grouped['average_user_satisfaction']\
       .apply(lambda x: f"${x['mean']*100:.0f}$ ±${x['std']*100:.0f}$", axis=1)
data_grouped['total_transformer_overload'] = data_grouped['total_transformer_overload']\
       .apply(lambda x: f"${x['mean']:.0f}$ ±${x['std']:.0f}$", axis=1)
data_grouped['total_reward'] = data_grouped['total_reward']\
       .apply(lambda x: f"${x['mean']:.1f}$ ±${x['std']:.1f}$", axis=1)
       
# drop the mean and std columns
data_grouped = data_grouped.droplevel(1, axis=1)
# drop duplicate columns
data_grouped = data_grouped.loc[:,~data_grouped.columns.duplicated()]
#rename columns
data_grouped.columns = ['Profits/Costs', 'Energy Charged',
                        'Energy Discharged', 'User Satisfaction',
                        'Transformer Overload',
                        'Reward']

print(data_grouped)
print(data_grouped.to_latex())

e
#%%

data = pd.read_csv('data_PST.csv')
#group by algotithm and get mean and std
columns = ['Unnamed: 0', 'run', 'Algorithm', 'total_ev_served', 'total_profits',
       'total_energy_charged', 'total_energy_discharged',
       'average_user_satisfaction', 'power_tracker_violation',
       'tracking_error', 'energy_tracking_error', 'energy_user_satisfaction',
       'total_transformer_overload', 'battery_degradation',
       'battery_degradation_calendar', 'battery_degradation_cycling',
       'total_reward']

columns_to_keep = ['Algorithm','total_energy_charged',
       'average_user_satisfaction',
       'tracking_error', 'energy_tracking_error',
       'total_reward']
data = data[columns_to_keep]

data_grouped = data.groupby('Algorithm').agg(['mean', 'std'])

# create new columns with the mean and std of the total_energy_charged combined as a string 
data_grouped['total_energy_charged'] = data_grouped['total_energy_charged']\
       .apply(lambda x: f"${x['mean']:.0f}$ ±${x['std']:.0f}$", axis=1)
data_grouped['average_user_satisfaction'] = data_grouped['average_user_satisfaction']\
       .apply(lambda x: f"${x['mean']*100:.0f}$ ±${x['std']*100:.0f}$", axis=1)
data_grouped['tracking_error'] = data_grouped['tracking_error']\
       .apply(lambda x: f"${x['mean']/1000:.1f}$ ±${(x['std']/1000):.1f}$", axis=1)
data_grouped['energy_tracking_error'] = data_grouped['energy_tracking_error']\
       .apply(lambda x: f"${x['mean']:.0f}$ ±${x['std']:.0f}$", axis=1)
data_grouped['total_reward'] = data_grouped['total_reward']\
       .apply(lambda x: f"${x['mean']/1000:.1f}$ ±${x['std']/1000:.1f}$", axis=1)
       
# drop the mean and std columns
data_grouped = data_grouped.droplevel(1, axis=1)
# print the results
# drop duplicate columns
data_grouped = data_grouped.loc[:,~data_grouped.columns.duplicated()]
#rename columns
data_grouped.columns = ['Energy Charged', 'User Satisfaction', 'Tracking Error', 'Energy Tracking Error', 'Reward']


#rename algorithm names with shorter names
#rename PowerTrackingErrorrMin to Optimal
# print(data_grouped)
print(data_grouped.to_latex())

