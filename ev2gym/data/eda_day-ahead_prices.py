import pandas as pd
import matplotlib.pyplot as plt

#read csv file Netherlands_day-ahead-2015-2023

# data = pd.read_csv(r'.\data\Netherlands_day-ahead-2015-2023.csv', sep=',', header=0)
# drop_columns = ['Country','Datetime (UTC)']
# data.drop(drop_columns, inplace=True, axis=1)
# data['year'] = pd.DatetimeIndex(data['Datetime (Local)']).year
# data['month'] = pd.DatetimeIndex(data['Datetime (Local)']).month
# data['day'] = pd.DatetimeIndex(data['Datetime (Local)']).day
# data['hour'] = pd.DatetimeIndex(data['Datetime (Local)']).hour
# print(data)


#read csv file c:\Users\sorfanoudakis\Downloads\pv_netherlands.csv

data = pd.read_csv(r'pv_netherlands.csv', sep=',', header=0)
print(data)

#drop time column
data.drop('time', inplace=True, axis=1)
#rename local_time to time
data.rename(columns={'local_time':'time'}, inplace=True)

desired_timescale = 10 ##### from env
simulation_length = 96 ##### from env
simulation_date = '2023-2-1 08:00:00' ##### from env
number_of_transformers = 3 ##### from env

dataset_timescale = 60
dataset_starting_date = '2022-01-01 00:00:00'


if desired_timescale > dataset_timescale:
    data = data.groupby(data.index // (desired_timescale/dataset_timescale)).max()
elif desired_timescale < dataset_timescale:
    # extend the dataset to data.shape[0] * (dataset_timescale/desired_timescale)
    # by repeating the data every (dataset_timescale/desired_timescale) rows
    data = data.loc[data.index.repeat(dataset_timescale/desired_timescale)].reset_index(drop=True)
    # data = data/ (dataset_timescale/desired_timescale)

# smooth data by taking the mean of every 5 rows
data['electricity'] = data['electricity'].rolling(window=60//desired_timescale,min_periods=1).mean()
# use other type of smoothing
data['electricity'] = data['electricity'].ewm(span=60//desired_timescale, adjust=True).mean()

data['electricity'] = data['electricity'] * 100

#plot the data for the first 96 rows
data['electricity'][: (60//desired_timescale * 24*7) ].plot()

plt.show()




# duplicate the data to have two years of data
data = pd.concat([data, data], ignore_index=True)

#select only 2022

#plot in subplots

# plt.figure(figsize=(10, 7))
# for i,year in enumerate(range(2015,2024)):
#     data_temp = data[data['year'] == year]
#     plt.subplot(3, 3, i+1)
#     plt.title(f'year {year}')
#     plt.xlabel('hour')
#     plt.ylabel('price')
    # plt.plot(data_temp['Datetime (Local)'], data_temp['Price (EUR/MWhe)'])
    
# save_plots
# plt.savefig('data/eda_day-ahead_prices.png')



    
        




