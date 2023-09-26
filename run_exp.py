"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

# run trai_DT.py in a tmux pane for each K and dataset

counter = 0
for K in [25, 50, 75, 100, 125, 150]:

    # for dataset in ['ddpg', 'random']:
    for dataset in ['random']:
        command = 'tmux new-session -d \; send-keys "python train_DT.py' + \
            ' --dataset ' + dataset + \
            ' --K ' + str(K) + \
            ' --device cuda:' + str(counter%2) + \
            ' --name K=' + str(K) + \
            '" Enter'
        os.system(command=command)    
        print(command)
        #wait for 20 seconds before starting the next experiment
        time.sleep(20)
        counter += 1
