from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible, RoundRobin_GF, RoundRobin_GF_off_allowed
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

config_path = "PublicPST_pr.yaml"
seed = 42

env = EV2Gym(config_file=config_path,                 
                verbose=False,
                save_plots=True,
                lightweight_plots= True)

power_setpoints = None

state, info = env.reset(seed=seed, power_setpoints=power_setpoints)

algorithm = RoundRobin_GF(env)
# algorithm = ChargeAsFastAsPossible(env)

done = False
while not done:
    action = algorithm.get_action(env)
    _, _, done, _, info = env.step(action)

metrics_to_keep = ['total_ev_served', 
                    'charging_costs',
                    'unmatched_power_costs',                    
                    'cpo_profits',
                    'total_energy_charged',
                    'energy_user_satisfaction',
                    'std_energy_user_satisfaction',
                    'min_energy_user_satisfaction',
                    'power_tracker_violation',
                    'tracking_error'
                    ]

data_gen_folder = './data_generation_results/'

if not os.path.exists(data_gen_folder):
    os.makedirs(data_gen_folder)

# Initialize charging transaction data list
charging_transaction_data = []

# print(f'self.EVs_profiles: {env.EVs}')
for ev_id, ev in enumerate(env.EVs_profiles):
    transaction = {
        'charger_id': ev.location,        
        'start_time': env.sim_starting_date + timedelta(minutes=ev.time_of_arrival * env.timescale),
        'end_time': env.sim_starting_date + timedelta(minutes=ev.time_of_departure * env.timescale),
        'energy_charged(kWh)': np.round(ev.battery_capacity - ev.battery_capacity_at_arrival, 2)
    }
    charging_transaction_data.append(transaction)
    
# Convert to DataFrame
charging_transaction_df = pd.DataFrame(charging_transaction_data)
charging_transaction_df.to_csv(data_gen_folder + 'charging_transaction_data.csv', index=False)

#Save charging prices to CSV
time_index = [env.sim_starting_date + timedelta(minutes=i * env.timescale) for i in range(env.simulation_length)]
charge_prices_df = pd.DataFrame(env.charge_prices[0], columns=['price'])
charge_prices_df.index = time_index
charge_prices_df.to_csv(data_gen_folder + 'charging_prices.csv', index=True)

#Save power setpoints to CSV
power_setpoints_df = pd.DataFrame(env.power_setpoints, columns=['power_setpoint'])
power_setpoints_df.index = time_index
power_setpoints_df.to_csv(data_gen_folder + 'power_setpoints.csv', index=True)

# create csv with the per charger load profile
load_profiles_df = pd.DataFrame(env.cs_power)
load_profiles_df.index = range(env.cs)
load_profiles_df.columns = time_index
load_profiles_df.to_csv(data_gen_folder + 'charger_load_profiles.csv', index=True)

print("\nSimulation Results:")
info = {key: info[key] for key in metrics_to_keep}
# print info dictionary with a for loop
for key, value in info.items():
    print(f"{key:<35}: {value:10.1f}")