from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import RoundRobin_GF, ChargeAsFastAsPossible
from ev2gym.utilities.utils import get_statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime





class EV2Gym_eval: 

    def __init__(self, eval_episodes, seed = 42):        
        
        config_path = "PublicPST_pr_eval.yaml"        
        
        # Initializing the simulator
        self.env = EV2Gym(config_file=config_path, 
                          seed=seed)

        # This is the default charging algorithm that maximizes charging fairness among EVs
        self.algorithm = RoundRobin_GF(self.env)

        # This is an alternative charging algorithm that charges EVs as fast as possible without considering the power setpoint.
        # algorithm = ChargeAsFastAsPossible(env)

        self.evaluation_day = 29
        self.eval_episodes = eval_episodes

    def get_raw_input(self):        
        
        power_setpoints = np.zeros(self.env.simulation_length)        
        _,_= self.env.reset(power_setpoints=power_setpoints)
        
        self.days_passed = 0
        self.done = False
                    
        while not done:
            time = self.env.sim_date
            hour = time.hour
            minutes = time.minute    
            
            if hour == 13 and minutes == 00:
                days_passed += 1
                if days_passed == self.evaluation_day:
                    print(f'Selecting Power Setpoints:', days_passed, time)
                    # set power setpoint for next day with a vector of random 96 values
                    start_idx = self.env.current_step # + 1
                    end_idx = min(start_idx + 96, self.env.simulation_length)                    
                    
                    ####
                    new_setpoints = np.random.uniform(low=30, high=100, size=end_idx - start_idx)
                    ####                    
                    assert len(new_setpoints) == 96, 
                    #### =================== ######## #### =================== ######## 
                    self.env.power_setpoints[start_idx:end_idx] = new_setpoints
                    #### =================== ######## #### =================== ######## 

                    # reset profit counters and all stats so evaluation-day metrics are clean
                    self.env.cpo_profits = 0.0
                    self.env.unmatched_power_costs = 0.0
                    # reset charger-level statistics
                    for cs in self.env.charging_stations:
                        cs.total_energy_charged = 0.0
                        cs.total_energy_discharged = 0.0
                        cs.charging_costs = 0.0
                        cs.total_evs_served = 0
                        cs.total_user_satisfaction = 0.0
                        cs.all_user_satisfaction = []                  
                
                if days_passed == (self.evaluation_day + 1):
                    # end_simulation and get info
                    print("Ending simulation.",days_passed, time)
                    self.env.done = True
                    info = get_statistics(self.env)
                    self.env.stats = info
                    done = True
                    break
                
            action = self.algorithm.get_action(self.env)
            _, _, done, _, info = self.env.step(action)

        print("\nDetailed Simulation Results:")
        print("-" * 70)
        print(f'Total EVs Served               : {info["total_ev_served"]:10.0f} EVs')
        print(f'Total Energy Charged           : {info["total_energy_charged"]:10.2f} kWh')
        print("-" * 70)
        # Show costs and profits
        print(f'Day-Ahead Energy Costs         : {info["charging_costs"]:10.2f} €')
        print(f'Unmatched Power Costs          : {info["unmatched_power_costs"]:10.2f} €')
        print(f'EV Charging Profits            : {info["cpo_profits"]:10.2f} €')
        print(f'Total profits of CPO           : {info["cpo_profits"] + info["unmatched_power_costs"] + info["charging_costs"]:10.2f} €')
        print("-" * 70)
        # The tracking error shows how well the power setpoints were tracked
        print(f'Tracking Error                 : {info["energy_tracking_error"]/4:10.2f} kWh')
        print("-" * 70)



    def plot_single_day(self):
        
        env = self.env
        # Plot only the last day's power setpoints and usage
        steps_per_day = int(24 * 60 / env.timescale)
        end_idx = env.current_step
        start_idx = max(0, end_idx - steps_per_day)
        last_day_range = pd.date_range(
            start=env.sim_starting_date + datetime.timedelta(minutes=env.timescale * start_idx),
            periods=end_idx - start_idx,
            freq=f"{env.timescale}min",
        )
        x_time = np.asarray(last_day_range.to_numpy())
        power_setpoints_last = np.asarray(env.power_setpoints, dtype=float).ravel()[start_idx:end_idx]
        power_usage_last = np.asarray(env.current_power_usage, dtype=float).ravel()[start_idx:end_idx]

        plt.figure(figsize=(6, 4))
        plt.step(x_time, power_setpoints_last, label='Power Setpoints (kW)', where='post')
        plt.step(x_time, power_usage_last, label='Actual Power Usage (kW)', where='post')
        plt.xlabel('Time')
        plt.ylabel('Power (kW)')
        plt.title('Last Day: Power Setpoints vs Actual Power Usage')
        tick_times = pd.date_range(start=last_day_range[0], end=last_day_range[-1], periods=8)
        plt.xticks(ticks=np.asarray(tick_times.to_numpy()), labels=[t.strftime('%H:%M') for t in tick_times], rotation=45)
        plt.legend()
        plt.grid(True, which='minor', axis='both')
        plt.tight_layout()
        plt.show()
        
        
if __name__ == "__main__":
    
    eval_days = 3
    
    evaluator = EV2Gym_eval(eval_episodes=eval_days)
    
    #Start of evaluation
    
    for t in eval_days:
        input_data_raw = evaluator.get_raw_input()
        
        
    