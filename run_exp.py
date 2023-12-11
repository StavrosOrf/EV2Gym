"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

# run trai_DT.py in a tmux pane for each K and dataset

batch_size = 128
num_steps_per_iter = 20000
max_iters = 2000

counter = 0
for K in [50,100]:
    # for dataset in ['ddpg', 'random']:
    for dataset in ["random"]:
        for embed_dim in [1024]:
            #   ' --device cuda:0' + str(counter % 2) + \
            for n_layer, n_head in [(3, 4)]: #(3, 1),
                command = 'tmux new-session -d \; send-keys "python train_DT.py' + \
                    ' --dataset ' + dataset + \
                    ' --K ' + str(K) + \
                    ' --device cuda:0' + \
                    ' --embed_dim ' + str(embed_dim) + \
                    ' --n_layer ' + str(n_layer) + \
                    ' --n_head ' + str(n_head) + \
                    ' --max_iters=' + str(max_iters) + \
                    ' --batch_size=' + str(batch_size) + \
                    ' --num_steps_per_iter=' + str(num_steps_per_iter) + \
                    ' --group_name ' + '"ModelSize_"' + \
                    ' --name K=' + str(K) + \
                    ',embed_dim=' + str(embed_dim) + \
                    ',n_layer=' + str(n_layer) +\
                    ',max_iters=' + str(max_iters) + \
                    ',num_steps_per_iter=' + str(num_steps_per_iter) + \
                    ',batch_size=' + str(batch_size) + \
                    ',batch_size=' + str(batch_size) + \
                    ',n_head=' + str(n_head) + \
                    '" Enter'
                os.system(command=command)
                print(command)
                # wait for 20 seconds before starting the next experiment
                time.sleep(20)
                counter += 1
