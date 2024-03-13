"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time
import yaml
import subprocess
from evaluator import evaluator

file_path = './EVsSimulator/example_config_files/V2G_MPC.yaml'

counter = 0
# for config in ['V2GProfitMax', 'PublicPST', 'V2GProfitPlusLoads']:
for cs in [5, 10, 20, 30, 40, 50, 75, 100, 125]: 
# for cs in [5, 10]: 
        #read the yaml file as yaml
        print("Number of charging stations: ", cs)
        file = yaml.load(open(file_path, 'r'), Loader=yaml.FullLoader)
        #change the number of charging stations
        file['number_of_charging_stations'] = cs
        #save the yaml file
        with open(file_path, 'w') as f:
            yaml.dump(file, f)
        
        evaluator()
        #run the experiment
        # command = "C:/Users/stayr/anaconda3/envs/l2rpn/python.exe evaluator.py"
        # print(command)
        # process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        # process.wait()
        # print(process.returncode)
        # os.system(command=command)
        
        counter += 1