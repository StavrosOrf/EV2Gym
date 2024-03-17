import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Plot total capacity lost (battery degradation) for each algorithm as a histogram for each run
data = pd.read_csv('C:/Users/stayr/Downloads/data_1.csv')
print(data.columns)

columns_to_keep = ['run', 'Algorithm', 'battery_degradation']

#change name of algorithms
data['Algorithm'] = data['Algorithm'].replace('ChargeAsFastAsPossibleToDesiredCapacity', 'As Fast As Possible')
data['Algorithm'] = data['Algorithm'].replace('ChargeAsFastAsPossible', 'As Fast As Possible')
data['Algorithm'] = data['Algorithm'].replace('OCCF_V2G', 'OCMF_V2G')
data['Algorithm'] = data['Algorithm'].replace('OCCF_G2V', 'OCMF_G2V')
data['Algorithm'] = data['Algorithm'].replace('eMPC_V2G', 'eMPC_V2G')
data['Algorithm'] = data['Algorithm'].replace('eMPC_G2V', 'eMPC_G2V')

# Divide battery degradation by total_ev_served 
data['battery_degradation'] = data['battery_degradation']/data['total_ev_served']

# use seaborn to plot the histogram
sns.set(style="white")

data = data[columns_to_keep]

plt.rc('font', family='serif')
# plot the histogram
# g = sns.FacetGrid(data, col="Algorithm",row='run')
ax = sns.histplot(data, x="battery_degradation", hue="Algorithm",
                  palette="Set1",
             stat="probability",
             common_norm=True,
             fill=True,
             kde=True,
             element="step",
             bins=40)

#change x title
plt.xlabel('Battery Degradation [%]',
           fontsize=22,           
           )

plt.ylabel('Probability',
              fontsize=22,              
              )

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
    
#change x ticks style to scientific notation
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#ijncrease x ticks font size
plt.xticks(fontsize=20)
#increase y ticks font size
plt.yticks(fontsize=20)
#ioncrease legend font size
plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='21') # for legend title
             
plt.show()
