import pandas as pd



data = pd.read_csv('data_PST.csv')
# data = pd.read_csv('data_Profitmax.csv')

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
# round up the values of certain columns
data_grouped = data_grouped.round({'total_energy_charged': 0,
                                   'average_user_satisfaction': 2,
                                   'tracking_error': 0, 
                                   'energy_tracking_error': 1,
                                   'total_reward': 0})

print(data_grouped)
