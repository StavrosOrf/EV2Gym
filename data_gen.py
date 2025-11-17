from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible, RoundRobin_GF, RoundRobin_GF_off_allowed

config_path = "PublicPST_pr.yaml"
n_runs = 1

seed = 42

for run in range(n_runs):
    env = EV2Gym(config_file=config_path,                 
                 verbose=False,
                 save_plots=True,
                 lightweight_plots= True)
    
    state, info = env.reset(seed=seed+run)
    algorithm = RoundRobin_GF(env, verbose=False)    
    # algorithm = RoundRobin_GF_off_allowed(env)    
    # algorithm = ChargeAsFastAsPossible(env)
    
    done = False
    while not done:
        action = algorithm.get_action(env)
        _, _, done, _, info = env.step(action)

    
    metrics_to_keep = ['total_ev_served', 
                       'total_profits',
                       'total_energy_charged',
                       'energy_user_satisfaction',
                       'std_energy_user_satisfaction',
                       'min_energy_user_satisfaction',
                       'power_tracker_violation',
                       'tracking_error'
                       ]
    
    info = {key: info[key] for key in metrics_to_keep}
    # print info dictionary with a for loop
    for key, value in info.items():
        print(f"{key:<35}: {value:10.1f}")