#%%
import pandas as pd

#%%
data = pd.read_csv('data_V2GProfitPlusLoads.csv')

columns_to_keep = ['Algorithm','total_profits',
       'total_energy_charged', 'total_energy_discharged',
       'average_user_satisfaction',
       'total_transformer_overload',
       'battery_degradation',
       'battery_degradation_calendar', 'battery_degradation_cycling',
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
data_grouped['battery_degradation'] = data_grouped['battery_degradation']\
       .apply(lambda x: f"${x['mean']*1000:.2f}$ ±${x['std']*1000:.2f}$", axis=1)
data_grouped['battery_degradation_calendar'] = data_grouped['battery_degradation_calendar']\
       .apply(lambda x: f"${x['mean']*1000:.2f}$ ±${x['std']*1000:.2f}$", axis=1)
data_grouped['battery_degradation_cycling'] = data_grouped['battery_degradation_cycling']\
       .apply(lambda x: f"${x['mean']*1000:.2f}$ ±${x['std']*1000:.2f}$", axis=1)
data_grouped['total_reward'] = data_grouped['total_reward']\
       .apply(lambda x: f"${x['mean']/1000:.1f}$ ±${x['std']/1000:.1f}$", axis=1)
       
# drop the mean and std columns
data_grouped = data_grouped.droplevel(1, axis=1)
# drop duplicate columns
data_grouped = data_grouped.loc[:,~data_grouped.columns.duplicated()]

#rename columns

data_grouped.columns = ['Profits/Costs', 'Energy Charged',
                            'Energy Discharged', 'User Satisfaction',
                            'Transformer Overload',
                            'Battery Degradation', 'Battery Degradation Calendar',
                            'Battery Degradation Cycling', 'Reward']

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

exit()
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

