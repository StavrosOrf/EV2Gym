import pickle
import numpy as np
import pandas as pd

#open pickle file and load the data
env_path = 'results/sim_2024_05_15_980717/env.pkl' #100 EVSE
env_path = 'results/sim_2024_05_15_555985/env.pkl' #50 EVSE
env_path = 'results/sim_2024_05_16_608373/env.pkl' #10 EVSE

env_path = 'results/sim_2024_05_16_374652/env.pkl' #100 EVSE
env_path = 'results/sim_2024_05_16_750638/env.pkl' #50 EVSE
env_path = 'results/sim_2024_05_16_433072/env.pkl' #10 EVSE


pickle_in = open(env_path, 'rb')

env = pickle.load(pickle_in)

print(env.cs_power.shape)

cs_power = env.cs_power

print(env.sim_date)
print(env.sim_starting_date)

print(f'max power: {np.max(cs_power)}')
print(f'min power: {np.min(cs_power)}')

# make a plot of the charging station power
import matplotlib.pyplot as plt


# numpy to pandas add a column with the time of the day
cs_power_df = pd.DataFrame(cs_power.T, columns=[f'cs_{i}' for i in range(cs_power.shape[0])])


#add a total power column
cs_power_df['total_power'] = cs_power_df.sum(axis=1)

sim_starting_date = env.sim_starting_date
cs_power_df['time'] = pd.date_range(sim_starting_date, periods=cs_power.shape[1], freq='15T')


#plot total power
plt.plot(cs_power_df['time'], cs_power_df['total_power'])
plt.xlabel('Time')
plt.ylabel('Total Power')
plt.title('Total Power vs Time')
plt.show()

# print df summary
print(cs_power_df.describe())


# reareange the columns
cols = cs_power_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
cols = cols[-1:] + cols[:-1]
cs_power_df = cs_power_df[cols]
print(cs_power_df.head())
# save the data to a csv file except the index column
cs_power_df.to_csv('10_EVSE_300kW_Limit.csv', index=False)



