"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time
import yaml
import subprocess
from evaluator import evaluator

file_path = './ev2gym/example_config_files/V2G_MPC.yaml'

counter = 0
for cs in [5,15]:
    #read the yaml file as yaml
    print("Number of charging stations: ", cs)
    file = yaml.load(open(file_path, 'r'), Loader=yaml.FullLoader)
    #change the number of charging stations
    file['number_of_charging_stations'] = cs
    #save the yaml file
    with open(file_path, 'w') as f:
        yaml.dump(file, f)

    evaluator()
    counter += 1

# counter = 0
# for m in [0.9, 1, 1.1, 1.2]:
#     # read the yaml file as yaml
#     print("Discharge price factor: ", m)
#     file = yaml.load(open(file_path, 'r'), Loader=yaml.FullLoader)
#     # change the number of charging stations
#     file['discharge_price_factor'] = m
#     # save the yaml file
#     with open(file_path, 'w') as f:
#         yaml.dump(file, f)

#     evaluator()
#     counter += 1
