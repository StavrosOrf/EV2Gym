# use seaborn for plot
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load_execution time data

# paths = ['results/eval_6cs_3tr_V2G_MPC_5_algos_20_exp_2024_03_07_610105',
#          #  'results/eval_48cs_3tr_V2G_MPC_5_algos_5_exp_2024_03_07_166424',
#          'results/eval_30cs_3tr_V2G_MPC_5_algos_2_exp_2024_03_08_380121',
#          'results/eval_18cs_3tr_V2G_MPC_5_algos_2_exp_2024_03_08_704132',
#          'results/eval_24cs_3tr_V2G_MPC_5_algos_5_exp_2024_03_07_055202',
#          'results/eval_12cs_3tr_V2G_MPC_5_algos_20_exp_2024_03_07_523345']

paths = ['results/eval_5cs_3tr_V2G_MPC_9_algos_2_exp_2024_03_13_334101',
         'results/eval_10cs_3tr_V2G_MPC_9_algos_2_exp_2024_03_13_451909',
            'results/eval_20cs_3tr_V2G_MPC_9_algos_2_exp_2024_03_13_502174',
            'results/eval_30cs_3tr_V2G_MPC_9_algos_2_exp_2024_03_13_336628',
            'results/eval_50cs_3tr_V2G_MPC_9_algos_1_exp_2024_03_14_420361',]            


order_list = [5, 10, 20, 30,50]
results = None
results_all = None

for path in paths:

    data_og = pd.read_csv(f'{path}/data.csv')
    #drop rows where algorithm = 'ChargeAsFastAsPossibleToDesiredCapacity'
    data_og = data_og[data_og['Algorithm'] != 'ChargeAsFastAsPossibleToDesiredCapacity']
    
    #drop rows where control horizon = -1
    data_og = data_og[data_og['control_horizon'] != -1]
    data_og = data_og[data_og['control_horizon'] != 30]
    
    # data_og['Algorithm'] = data_og['Algorithm'] + \
    #     data_og['control_horizon'].astype(str)
        
    # group by algorithm and find mean and std time
    data = data_og.groupby(['Algorithm']).agg(
        {'time': ['mean', 'std']}).reset_index()
    cs_number = path.split('_')[1].split('cs')[0]
    print(f'CS number: {cs_number}')

    # create new dataframe to store algorithm, cs_number, mean and std time
    new_data = pd.DataFrame(
        columns=['Algorithm', 'cs_number', 'mean_time', 'std_time'])
    new_data['Algorithm'] = data['Algorithm']
    new_data['cs_number'] = cs_number
    new_data['mean_time'] = data['time']['mean']
    new_data['std_time'] = 5  # data['time']['std']

    data_og['cs_number'] = int(cs_number)
    data_og['time'] = data_og['time'] / 96

    if results is None:
        results = new_data
        results_all = data_og
    else:
        results = pd.concat([results, new_data])
        #divide by 9600 to get the time per step
        
        results_all = pd.concat([results_all, data_og])

# print(results)
print(results_all)
# sort the results by cs_number
results = results.sort_values(by=['cs_number'])
# cahnge algorithm names
results_all['Algorithm'] = results_all['Algorithm'].replace({'ChargeAsFastAsPossibleToDesiredCapacity': 'AFAP',
                                                             'ChargeAsFastAsPossible': 'AFAP',
                                                             'ChargeAsLateAsPossible': 'ALAP',
                                                             'OCCF_V2G': 'OCMF_V2G',
                                                             'OCCF_G2V': 'OCMF_G2V',})

print(results_all.columns)
#drop rows where algorithm = 'ChargeAsFastAsPossibleToDesiredCapacity'

results_all = results_all[results_all['Algorithm'] != 'AFAP']


# plot in the same plot the execution time of the algorithms for different cs numbers
sns.set_theme(style="whitegrid")
# change font style serif
plt.rcParams["font.family"] = "serif"

# use tex for font
# plt.rc('text', usetex=True)

# ax = sns.barplot(x="cs_number", y="mean_time",
#                  hue="Algorithm",
#                  data=results)
# use catplots and connect the means sorted by cs_number

ax = sns.catplot(x="cs_number",
                 y="time",
                 hue="Algorithm",
                 data=results_all,
                 native_scale=True,
                 kind="point",
                 errorbar="se",
                 order=order_list,
                 height=6,
                 linestyle=['--', '--', '-.', '-.'],
                 markers=['o', 's', 'D', 'v', 'p', 'P', '*', 'X', 'h'],
                 aspect=.75,
                 alpha=0.7,
                 legend_out=False,)




# logarithmic scale
# ax.set(yscale="log")
# use xticks to match the cs_number and real number of charging stations
# ax.set(xticks=np.arange(5,51, step=5),
#        xticklabels=np.arange(5, 51, step=5))

# use x ticks only for the x values in order_list
ax.set(xticks=order_list,
       xticklabels=order_list)

#increase font size of legend
plt.legend(fontsize=22)
# ax._legend.remove()

ax.set(yticks=np.arange(0, 251, step=50),
       yticklabels=np.arange(0, 251, step=50))
# set labels
#increase font size
ax.set_xticklabels(fontsize=22)
ax.set_yticklabels(fontsize=22)
ax.set_xlabels("Charging Stations", fontsize=22)
ax.set_ylabels("Execution Time (s)", fontsize=22)

# ax = ax.set_axis_labels("Charging Stations", "Execution time (s)")
# move legend inside the plot

plt.show()
exit()
# use catplot to plot profits
ax = sns.catplot(x="cs_number",
                 y="total_profits",
                 hue="Algorithm",
                 data=results_all,
                 kind="point",
                 order=order_list,
                 height=6,
                 aspect=.75,
                 legend_out=False,)
# set labels
ax = ax.set_axis_labels("Charging Stations", "Profit")

plt.show()
