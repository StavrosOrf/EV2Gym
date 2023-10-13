# #write script to read in data and do some basic EDA for the elaad data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# #print all file in directory
# for dirname, _, filenames in os.walk('.\data'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # df_arrival_week = pd.read_csv('.\data\distribution-of-arrival.csv') #weekdays
# # df_arrival_weekend = pd.read_csv('.\data\distribution-of-arrival-weekend.csv') #weekends
# # df_connection_time = pd.read_csv('.\data\distribution-of-connection-time.csv') #connection time
# # df_denergy_demand = pd.read_csv('.\data\distribution-of-energy-demand.csv') #energy demand

# print(df_arrival_week['public'].sum())
# print(df_arrival_weekend['public'].sum())
# print(df_arrival_week)
# print(df_arrival_weekend)

# date = datetime.datetime(2021, 1, 1, 0, 0, 0)

# arrival_times = []
# energy_demand = []
# connection_time = []

# numbers = np.random.randint(0,100,size=1)

# for i in numbers:
#     energy_demand.append(df_denergy_demand['public'].iloc[i])
#     connection_time.append(df_connection_time['public'].iloc[i])
# plt.hist(energy_demand, bins=20)
# plt.show()
# plt.hist(connection_time, bins=24)
# for k in range(3650):
#     # for i in range(int(24*60/15)):    
#     #     date += datetime.timedelta(minutes=15)

#         # print(date)
#         #this is how I am gonna sample public ev spawns
#         #Allow to choose residential, public, workplace scenarios
#         # if np.random.rand(1)*100 < df_arrival_week['public'].iloc[i]:
#         #     arrival_times.append(i)

    
        
# # plot histogram of arrival times
# plt.hist(arrival_times, bins=96)
# plt.show()

# #read in data from xlsx file
df = pd.read_excel('.\data\elaadnl_open_ev_datasets.xlsx',
                    sheet_name='open_transactions')
#get all keys
keys = df.keys()

df['hour_of_arrival'] = df['UTCTransactionStart'].dt.hour

#drop all other columns except for "ChargeTime" and "UTCTransactionStart"
for k in keys:
    if k != "ChargeTime" and k != "hour_of_arrival":
        df = df.drop(k, axis=1)


print(df.head())

#plot a histogram of the hour_of_arrival vs the Chargetime for every hour in the same figure
# also add legends and labels, and make plots transparent
# plt.figure()
#save the bins and edges for later use

data_table = np.zeros((24,48))

for i in range(24):
    bins, edges = np.histogram(df['ChargeTime'].loc[df['hour_of_arrival'] == i], bins=48)
    #normalize the bins
    bins = bins/bins.sum()
    data_table[i,:] = bins

    #smooth the data
    bins = np.convolve(bins, np.ones(5)/5, mode='same')


    #plot cdf    
    # plt.plot(edges[:-1], np.cumsum(bins), label=f'Hour {i}')

    plt.plot(edges[:-1], bins, label=f'Hour {i}')
plt.legend()
plt.xlabel('Charge Time')
plt.ylabel('Frequency')
plt.title('Charge Time vs Hour of Arrival')
plt.show()

#save data_table as numpy array
np.save('time_of_connection_vs_hour.npy', data_table)
