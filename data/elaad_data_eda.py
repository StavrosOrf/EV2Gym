#write script to read in data and do some basic EDA for the elaad data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

#print all file in directory
for dirname, _, filenames in os.walk('.\data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df_arrival_week = pd.read_csv('.\data\distribution-of-arrival.csv') #weekdays
df_arrival_weekend = pd.read_csv('.\data\distribution-of-arrival-weekend.csv') #weekends
df_connection_time = pd.read_csv('.\data\distribution-of-connection-time.csv') #connection time
df_denergy_demand = pd.read_csv('.\data\distribution-of-energy-demand.csv') #energy demand

print(df_arrival_week['public'].sum())
print(df_arrival_weekend['public'].sum())
print(df_arrival_week)
print(df_arrival_weekend)
# print(df_connection_time)
# print(df_denergy_demand)

# df_arrival_week.plot(x='hour', y=['public', 'private', 'workplace'], kind='bar', figsize=(10,8))
#     Percentage of charging events  private  public  workplace
# 0                                0     72.0    72.0       72.0
# 1                                1     56.5    44.9       46.4
# 2                                2     43.0    38.8       25.7
# 3                                3     38.8    29.1       20.1
# 4                                4     35.4    23.7       16.6
# ..                             ...      ...     ...        ...
# 96                              96      0.8     0.3        0.3
# 97                              97      0.7     0.2        0.1
# 98                              98      0.6     0.1        0.0
# 99                              99      0.5     0.0        0.0
# 100                            100      0.0     0.0        0.0

date = datetime.datetime(2021, 1, 1, 0, 0, 0)

arrival_times = []
energy_demand = []
connection_time = []

numbers = np.random.randint(0,100,size=1000000)

for i in numbers:
    energy_demand.append(df_denergy_demand['public'].iloc[i])
    connection_time.append(df_connection_time['public'].iloc[i])
plt.hist(energy_demand, bins=20)
plt.show()
plt.hist(connection_time, bins=24)
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
plt.show()

# #read in data from xlsx file
# df = pd.read_excel('.\data\elaadnl_open_ev_datasets.xlsx',
#                     sheet_name='open_transactions')

# print(df.head())

# #create a new column for hour of arrival and departure
# df['hour_of_arrival'] = df['UTCTransactionStart'].dt.hour
# df['hour_of_departure'] = df['UTCTransactionStop'].dt.hour

# #create a new column for day of arrival and departure
# df['day_of_arrival'] = df['UTCTransactionStart'].dt.day
# df['day_of_departure'] = df['UTCTransactionStop'].dt.day

# #sample from  df['hour_of_arrival'] to get a sense of the distribution
# print(f'Hour of arrival: {df["hour_of_arrival"].sample(10)}')
# print(f' Charge time: {df["ChargeTime"].sample(10)}')

# print(df.nunique())

# #plot histogram of uniqe chargepoint_IDs of the sum of the charge time
# df.groupby('ChargePoint').hist(bins=850, figsize=(10,8))
# plt.show()


# #check for outliers
# print(df.describe())

# #plot histograms
# df.hist(bins=31, figsize=(10,8))
# # plt.show()

# exit()
# df = pd.read_excel('.\data\elaadnl_open_ev_datasets.xlsx',
#                     sheet_name='open_metervalues')

# print(df.head())
# print(df.info())

# #check for missing values
# print(df.isnull().sum())

# #check for duplicates
# print(df.duplicated().sum())

# #check for outliers
# print(df.describe())

# #plot histograms
# df.hist(bins=50, figsize=(10,8))
# plt.show()