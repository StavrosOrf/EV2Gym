import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

paths = ['C:/Users/stayr/OneDrive - Delft University of Technology/Paper Experiments/MPC/results/eval_30cs_3tr_V2G_MPC_5_algos_30_exp_2024_03_17_776538/data.csv',
         'C:/Users/stayr/OneDrive - Delft University of Technology/Paper Experiments/MPC/results/eval_30cs_3tr_V2G_MPC_5_algos_30_exp_2024_03_17_445097/data.csv',
         'C:/Users/stayr/OneDrive - Delft University of Technology/Paper Experiments/MPC/results/eval_30cs_3tr_V2G_MPC_5_algos_30_exp_2024_03_17_185972/data.csv',
         'C:/Users/stayr/OneDrive - Delft University of Technology/Paper Experiments/MPC/results/eval_30cs_3tr_V2G_MPC_5_algos_30_exp_2024_03_17_263522/data.csv',
         'C:/Users/stayr/OneDrive - Delft University of Technology/Paper Experiments/MPC/results/eval_30cs_3tr_V2G_MPC_5_algos_30_exp_2024_03_17_474691/data.csv',
         ]

price_multipliers = [0.8,0.9,1,1.1,1.2]
plt.rc('font', family='serif')


for i, path in enumerate(paths):

    data = pd.read_csv(f'{path}')
    data['Algorithm'] = data['Algorithm'].replace('ChargeAsFastAsPossibleToDesiredCapacity', 'As Fast As Possible')
    
    plt.subplot(5, 1, i+1)    
    ax = sns.histplot(data, x="total_profits", hue="Algorithm", #palette="Set1",
             stat="probability",
             common_norm=True,
             fill=True,
            #  kde=True,
            #  kde_kws={
            #               'linestyle':'--'},
             element="step",             
             bins=35)
    
    # colors = sns.color_palette("Set1")
    #get colors of ax
    # colors = ax.get_lines()
    
    
    # colors = np.linspace(0, 1, 5)
    # colors = [plt.cm.Set1(x) for x in colors]
    for j, algo in enumerate(data['Algorithm'].unique()):
        sns.kdeplot(data[data['Algorithm'] == algo]['total_profits'], linestyle='--',
                    )

    plt.title(f'$m={price_multipliers[i]}$',
              fontsize=20,)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=17)
    
    if i == 4:
        plt.xticks(np.arange(-200, 200, 50),
                   np.arange(-200, 200, 50),
                   fontsize=20)
    else:
        plt.xticks(np.arange(-200, 200, 50),
                   " " * len(np.arange(-200, 200, 50)),
                   fontsize=20)

    plt.xlim(-200, 200)
    if i ==2:
        plt.ylabel('Probability',
               fontsize=22,              
               )
    else:
        plt.ylabel('',
               fontsize=22,              
               )
        
    #change x title
    if i == 4:        
        plt.xlabel('Total Costs/ Profits (€)',
            fontsize=22)
    else:
        plt.xlabel('',
            fontsize=22)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    handles = ax.get_legend().legend_handles
    # if i == 0:                               
    #     # print(ax.get_legend().handles)
    #     pass
    #     ax.legend(
    #         handles=ax.get_legend().legendHandles,
    #         labels=["As Fast As Possible",
    #                 'OCMF V2G',
    #                 'OCMF G2V',
    #                 'eMPC V2G',
    #                 'eMPC G2V'],
    #         fontsize=22,
    #         loc='upper right', ncol=2, title="")
    #     # # plt.legend(loc='upper left')
    #     # plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
    #     # plt.setp(ax.get_legend().get_title(), fontsize='21') # for legend title
    #     # # set ncols=2 to have 2 columns in the legend
    #     # plt.setp(ax.get_legend().get_texts(),ncols=2)
        
    # else:
    plt.legend([],[], frameon=False)

plt.legend( handles=handles,
            labels=["As Fast As Possible",
                    'OCMF V2G',
                    'OCMF G2V',
                    'eMPC V2G',
                    'eMPC G2V'],
            loc='upper center',
           bbox_to_anchor=(0.5, -0.8),
           alignment='right',
           fancybox=True,
           shadow=True,
           ncol=3,
           fontsize=21)    
plt.show()    
    
