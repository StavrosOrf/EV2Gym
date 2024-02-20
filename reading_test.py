import pandas as pd
import pickle
import os
import EVsSimulator.models.replay

# Load the pickle file

pickle_file_path = 'C:/Users/Emre/OneDrive/Masaüstü/Drive2X/Codes/Repo/EVsSimulator/replay/3cs_1tr/replay_sim_48_2024-02-04_18-55-15-286380_replay.pkl'

with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Inspect the loaded data
print(data) # Print the data
print(dir(EVsSimulator.models.replay.EvCityReplay))  # List all attributes and methods
print(data.stats) # Print the statistics

