"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import sys
import time



#run trai_DT.py in a tmux pane
# os.system("tmux send-keys -t 0 'python3 train_DT.py' Enter")
os.system('tmux new-session -d -s my_session \; send-keys "python3 test.py --" Enter')
