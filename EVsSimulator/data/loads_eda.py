# This script loads the residential_loads.csv file and plots the data.


import pandas as pd

# Print the first few rows of the dataframe
import matplotlib.pyplot as plt
# Plot the data using pandas
# data.plot()

# Load the data
data = pd.read_csv('residential_loads.csv',header=None)


desired_timescale = 15 ##### from env
simulation_length = 96 ##### from env
simulation_date = '2023-2-1 08:00:00' ##### from env
number_of_transformers = 3 ##### from env

dataset_timescale = 15
dataset_starting_date = '2022-01-01 00:00:00'


if desired_timescale > dataset_timescale:
    data = data.groupby(data.index // (desired_timescale/dataset_timescale)).max()
elif desired_timescale < dataset_timescale:
    # extend the dataset to data.shape[0] * (dataset_timescale/desired_timescale)
    # by repeating the data every (dataset_timescale/desired_timescale) rows
    data = data.loc[data.index.repeat(dataset_timescale/desired_timescale)].reset_index(drop=True)
    # data = data/ (dataset_timescale/desired_timescale)

# duplicate the data to have two years of data
data = pd.concat([data, data], ignore_index=True)

#add a date column to the dataframe
data['date'] = pd.date_range(start=dataset_starting_date, periods=data.shape[0], freq=f'{desired_timescale}T')
    
#find year of the data
year = int(dataset_starting_date.split('-')[0])
# replace the year of the simulation date with the year of the data
simulation_date = f'{year}-{simulation_date.split("-")[1]}-{simulation_date.split("-")[2]}'
print(f'simulation date: {simulation_date}')
simulation_index = data[data['date'] == simulation_date].index[0]

#select the data for the simulation date
data = data[simulation_index:simulation_index+simulation_length]

print(f'simulation index: {simulation_index}')
print(data.head())
#drop the date column
data = data.drop(columns=['date'])

print(data.columns)
print(data.shape)

for i in range(number_of_transformers):
    data['tr_'+str(i)] = data.sample(10, axis=1).sum(axis=1)

#plot data[tr_ data
data['tr_0'].plot()
data['tr_1'].plot()
data['tr_2'].plot()

#show the time of the simulation on the x-axis

plt.xlabel('Time')
plt.ylabel('Power (kW)')

# Show the plot
plt.show()

exit()


#select 7 random columns and sum them
data['sum'] = data.sample(10, axis=1).sum(axis=1)
# Plot the data using pandas
# data['sum'].plot()

#plot the daily sum of the data (aggregate every 96 rows)



data['sum'][:96].plot()


print(data.columns)
print(data.shape)

# Show the plot
plt.show()