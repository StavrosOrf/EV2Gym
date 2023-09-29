"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

# run trai_DT.py in a tmux pane for each K and dataset

counter = 0
for K in [25,150]:

    # for dataset in ['ddpg', 'random']:
    for dataset in ['random']:
        for embed_dim in [128, 256, 512]:
            for n_layer, n_head in [(3, 1), (4, 3)]:
                command = 'tmux new-session -d \; send-keys "python train_DT.py' + \
                    ' --dataset ' + dataset + \
                    ' --K ' + str(K) + \
                    ' --device cuda:' + str(counter % 2) + \
                    ' --embed_dim' + str(embed_dim) + \
                    ' --embed_dim' + str(n_layer) + \
                    ' --embed_dim' + str(n_head) + \
                    ' --name K=' + str(K) + \
                    ',embed_dim=' + str(embed_dim) + \
                    ',n_layer=' + str(n_layer) +\
                    ',n_head=' + str(n_head) + \
                    '" Enter'
                os.system(command=command)
                print(command)
                # wait for 20 seconds before starting the next experiment
                time.sleep(20)
                counter += 1
