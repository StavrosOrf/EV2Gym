'''
This script is used to load the EVs of the simulation and plot the behavior of the EVs.
'''

import pandas as pd

import os
import sys
import datetime
sys.path.append(os.path.realpath('../'))

from EVsSimulator.models.ev import EV
import datetime
# load from replay fil

replay_path = "../replay/replay_sim_2024_02_09_543405.pkl"
replay_path = "../replay/replay_sim_2024_02_10_957926.pkl" # test with this

replay_path = "../replay/replay_sim_2024_02_10_381485.pkl"
# replay_path = "../replay/replay_sim_2024_02_10_577632.pkl"
replay = pd.read_pickle(replay_path)
evs = replay.EVs

print(len(evs))

# gather the arrival and departure times of the EVs

data = pd.DataFrame(columns=['id', 'arrival', 'departure', 'required_energy'])

sim_date = replay.sim_date

print(sim_date)
for i,ev in enumerate(evs):
    required_energy = ev.battery_capacity - ev.battery_capacity_at_arrival
    # turn time into minutes using the timescale and the sim_date
    arrival = sim_date + \
        datetime.timedelta(minutes=replay.timescale) * ev.time_of_arrival

    departure = sim_date + \
        datetime.timedelta(minutes=replay.timescale) * ev.time_of_departure
    
    #encode the time as a string using the day of the week and the time of the day (30 min intervals)
    arrival = str(arrival.weekday()) + arrival.strftime('%H:%M')
    departure = str(departure.weekday()) + departure.strftime('%H:%M')
    
    new_row = pd.DataFrame({'id': [i], 'arrival': [arrival],
                            'departure': [departure], 'required_energy': [required_energy]})
    data = data._append(new_row, ignore_index=True)

print(data)

# plot histogram of the required energy vs the arrival time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

num_bins = 7*96
font_size = 25

for i in range(num_bins):
    arrival = sim_date + datetime.timedelta(minutes=replay.timescale) * i
    arrival = str(arrival.weekday()) + arrival.strftime('%H:%M')
    
    departure = sim_date + datetime.timedelta(minutes=replay.timescale) * i
    departure = str(departure.weekday()) + departure.strftime('%H:%M')
    
    new_row = pd.DataFrame({'id': [i], 'arrival': [arrival],
                            'departure': [departure], 'required_energy': [0]})
    data = data._append(new_row, ignore_index=True)
    
    
    
    


fig, ax = plt.subplots()

# plt.figure(figsize=(20, 10))
# plt.style.use('seaborn-darkgrid')
plt.grid(True, which='minor', axis='both')
plt.rcParams.update({'font.size': font_size})
plt.rcParams['font.family'] = ['serif']

# fig, ax = plt.subplots()



# ax.hist(data['arrival'], bins=num_bins, weights=data['required_energy'])
# ax.set_ylabel('Required Energy [kWh]')
# ax.set_xlabel('Time of Arrival')
# ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# plt.show()

# plot histogram of the normalized arrival time and departure time in the same plot
#sort data by arrival time
data = data.sort_values(by='arrival')

bin = data['arrival'].hist(bins=num_bins,density=True,label='Arrival Time',ax=ax)
# #print bin edges

data = data.sort_values(by='departure')
bin = data['departure'].hist(bins=num_bins,density=True,label='Departure Time',ax=ax)

ax.set_xticks([i for i in range(0,7*96+1,96 )])
# plt.xticks(rotation=45)
ax.set_ylabel('Frequency')
ax.set_xlabel('Time')


ax.legend()
plt.show()
