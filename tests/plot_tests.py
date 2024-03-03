import os 
import sys
sys.path.append(os.getcwd())


from EVsSimulator.vizuals.evaluator_plot import plot_total_power, plot_comparable_EV_SoC
from EVsSimulator.vizuals.evaluator_plot import plot_total_power_V2G, plot_actual_power_vs_setpoint
from EVsSimulator.vizuals.evaluator_plot import plot_comparable_EV_SoC_single, plot_prices

# plot_total_power_V2G(results_path='E:/GitHub/EVsSimulator/results/eval_5cs_1tr_V2G_MPC_5_algos_1_exp_2024_03_03_727260/plot_results_dict.pkl',
#                      save_path='E:/GitHub/EVsSimulator/results/eval_5cs_1tr_V2G_MPC_5_algos_1_exp_2024_03_03_727260/',
#                      algorithm_names=['Charge As Fast As Possible', 'OCCF V2G', 'OCCF G2V', 'eMPC V2G', 'eMPC G2V'])
plot_comparable_EV_SoC_single(results_path='E:/GitHub/EVsSimulator/results/eval_5cs_1tr_V2G_MPC_5_algos_1_exp_2024_03_03_727260/plot_results_dict.pkl',
                     save_path='E:/GitHub/EVsSimulator/results/eval_5cs_1tr_V2G_MPC_5_algos_1_exp_2024_03_03_727260/',
                     algorithm_names=['Charge As Fast As Possible', 'OCCF V2G', 'OCCF G2V', 'eMPC V2G', 'eMPC G2V'])
plot_prices(results_path='E:/GitHub/EVsSimulator/results/eval_5cs_1tr_V2G_MPC_5_algos_1_exp_2024_03_03_727260/plot_results_dict.pkl',
                     save_path='E:/GitHub/EVsSimulator/results/eval_5cs_1tr_V2G_MPC_5_algos_1_exp_2024_03_03_727260/',
                     algorithm_names=['Charge As Fast As Possible', 'OCCF V2G', 'OCCF G2V', 'eMPC V2G', 'eMPC G2V'])