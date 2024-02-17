"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

counter = 0
for algorithm in ['ddpg', 'td3', 'sac', 'a2c', 'ppo', 'tqc', 'trpo', 'ars', 'rppo']:
    command = 'tmux new-session -d \; send-keys "python3 train_stable_baselines.py' + \
        ' --algorithm ' + algorithm + \
        ' --device cuda:' + str(counter % 2) + \
        '" Enter'
    os.system(command=command)
    print(command)
    # wait for 10 seconds before starting the next experiment
    time.sleep(10)
    counter += 1
