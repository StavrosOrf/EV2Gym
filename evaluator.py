#This script reads the replay files and evaluates the performance.

import yaml
import os
import pickle
from utils.arg_parser import arg_parser
import numpy as np
from EVsSimulator import ev_city

args = arg_parser()
config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)

number_of_charging_stations = config["number_of_charging_stations"]
n_transformers = config["number_of_transformers"]
steps = config["simulation_length"]
timescale = config["timescale"]

n_test_cycles = 2#args.n_test_cycles


eval_replay_path = "./replay/" + \
        f'{number_of_charging_stations}cs_{n_transformers}tr/'
        
eval_replay_files = [f for f in os.listdir(
    eval_replay_path) if os.path.isfile(os.path.join(eval_replay_path, f))]
assert len(
    eval_replay_files) > 0, "No replay files found in evaluation replay folder"

print(f'Found {len(eval_replay_files)} replay files in {eval_replay_path}')
if n_test_cycles > len(eval_replay_files):
    n_test_cycles = len(eval_replay_files)

print(f'Number of test cycles: {n_test_cycles}')

# Load the replay files and aggregate the statistics
opt_energy_charged = []
opt_tracking_errors = []
opt_tracking_surpluses = []
opt_energy_user_satisfaction = []

for i in range(n_test_cycles):    
    replay = pickle.load(open(eval_replay_path + eval_replay_files[i], 'rb'))
    opt_energy_charged.append(replay.stats["total_energy_charged"])
    opt_tracking_errors.append(replay.stats["tracking_error"])
    opt_tracking_surpluses.append(replay.stats["power_tracker_violation"])
    opt_energy_user_satisfaction.append(replay.stats["energy_user_satisfaction"])
    

energy_charged = []
tracking_errors = []
tracking_surpluses = []
energy_user_satisfaction = []

for i in range(n_test_cycles):    
    replay_path = eval_replay_path + eval_replay_files[i]

    env = ev_city.EVCity(config_file = "config_files/config.yaml",    
                         load_from_replay_path=replay_path,                                                            
                         generate_rnd_game=False,                                 
                         render_mode=False,       
                         save_plots=False,
                         save_replay=False,                  
                         )
    state = env.reset()

    rewards = []    

    for i in range(env.simulation_length):        
        # all ports are charging instantly
        actions = np.ones(env.number_of_ports)

        new_state, reward, done, stats = env.step(
            actions, visualize=False)  # takes action
        rewards.append(reward)


        # input("Press Enter to continue...")        
        if done:
            # print(f'End of simulation at step {env.current_step}')
            break
    
    energy_charged.append(stats["total_energy_charged"])
    tracking_errors.append(stats["tracking_error"])
    tracking_surpluses.append(stats["power_tracker_violation"])
    energy_user_satisfaction.append(stats["energy_user_satisfaction"])

#Print the average and the standard deviation of the statistics
print(f'--------- Optimal -------------')
print(f'Average energy charged: {sum(opt_energy_charged)/len(opt_energy_charged)}',f'Standard deviation: {np.std(opt_energy_charged)}')
print(f'Average tracking error: {sum(opt_tracking_errors)/len(opt_tracking_errors)}',f'Standard deviation: {np.std(opt_tracking_errors)}')
print(f'Average power tracker violation: {sum(opt_tracking_surpluses)/len(opt_tracking_surpluses)}',f'Standard deviation: {np.std(opt_tracking_surpluses)}')
print(f'Average energy user satisfaction: {sum(opt_energy_user_satisfaction)/len(opt_energy_user_satisfaction)}',f'Standard deviation: {np.std(opt_energy_user_satisfaction)}')

#Print the average and the standard deviation of the statistics
print(f'--------- Unstirred -------------')
print(f'Avg. energy charged: {sum(energy_charged)/len(energy_charged)}',f'Std: {np.std(energy_charged)}')
print(f'Avg. tracking error: {sum(tracking_errors)/len(tracking_errors)}',f'Std: {np.std(tracking_errors)}')
print(f'Avg. power tracker violation: {sum(tracking_surpluses)/len(tracking_surpluses)}',f'Std: {np.std(tracking_surpluses)}')
print(f'Avg. energy user satisfaction: {sum(energy_user_satisfaction)/len(energy_user_satisfaction)}',f'Std: {np.std(energy_user_satisfaction)}')
