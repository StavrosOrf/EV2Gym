import pandas as pd
import matplotlib.pyplot as plt

#read csv file Netherlands_day-ahead-2015-2023

data = pd.read_csv(r'.\data\Netherlands_day-ahead-2015-2023.csv', sep=',', header=0)
drop_columns = ['Country','Datetime (UTC)']
data.drop(drop_columns, inplace=True, axis=1)
data['year'] = pd.DatetimeIndex(data['Datetime (Local)']).year
data['month'] = pd.DatetimeIndex(data['Datetime (Local)']).month
data['day'] = pd.DatetimeIndex(data['Datetime (Local)']).day
data['hour'] = pd.DatetimeIndex(data['Datetime (Local)']).hour
print(data)

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



    
        




