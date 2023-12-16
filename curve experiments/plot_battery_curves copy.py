#here we will plot the data


import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import pylab as pl
import matplotlib.dates as mdates

# df_dis = pd.read_csv('dis_honda_3.csv', sep=',')
df_dis = pd.read_csv('ch_honda_5.csv', sep=',')
#plot the current
df_dis = df_dis.iloc[:-500]

df_dis["current_setpoint"] = df_dis["AC_Power_set_ID2"]*1000 / df_dis["Output_Voltage_ID2"]

# -10 *\
    #  1000 / 394  # df[f"Output_Voltage_ID{port}"]
print(df_dis.columns)
columns_to_plot_dis = ["Output_Current_ID2",
                       "current_setpoint",      
                    "AC_Power_set_ID2", 
                    "Output_Voltage_ID2",                
                       'Capacity_EV_ID2',
                       'SoC_ID2',
                       ]

columns_to_plot_dis = ["SoC_ID2", "Linear Model", "Two Stage Model"]
df_dis['Linear Model'] =  0
df_dis['Linear Model'].at[0] = df_dis['SoC_ID2'].iloc[0]

df_dis['Two Stage Model'] = 0
df_dis['Two Stage Model'].at[0] = df_dis['Linear Model'].iloc[0]

df_dis.epoch = pd.to_datetime(df_dis.epoch, unit="ms")
print(df_dis.epoch)
df_dis.reset_index(inplace=True)
timestep = df_dis.epoch.iloc[1] - df_dis.epoch.iloc[0]
#turn timestep into minutes
timestep = timestep.total_seconds()/60
print(f'timestep: {timestep} minutes')


def two_stage_model(amps,voltage,timescale,max_battery_capacity,threshold,soc):
    pilot = amps
    voltage = voltage * math.sqrt(3)
    period = timescale
    # All calculations are done in terms of battery SoC, so we
    # convert pilot signal and max power into pilot and max rate of
    # change of SoC.
    pilot_dsoc = pilot * voltage / 1000 / \
        max_battery_capacity / (60 / period)
    max_dsoc = 14.5 * voltage / 1000 / \
        max_battery_capacity / (60 / period)

    if pilot_dsoc > max_dsoc:
        pilot_dsoc = max_dsoc

    # The pilot SoC rate of change has a new transition SoC at
    # which decreasing of max charging rate occurs.
    pilot_transition_soc = threshold + (
        pilot_dsoc - max_dsoc
    ) / max_dsoc * (threshold - 1)

    # The charging equation depends on whether the current SoC of
    # the battery is above or below the new transition SoC.
    if soc < pilot_transition_soc:
        # In the pre-rampdown region, the charging equation changes
        # depending on whether charging the battery over this
        # time period causes the battery to transition between
        # charging regions.
        if 1 <= (pilot_transition_soc - soc) / pilot_dsoc:
            curr_soc = pilot_dsoc + soc
        else:
            print(f'soc {soc}')
            curr_soc = 1 + np.exp(
                (pilot_dsoc + soc - pilot_transition_soc)
                / (pilot_transition_soc - 1)
            ) * (pilot_transition_soc - 1)
    else:
        curr_soc = 1 + np.exp(pilot_dsoc / (pilot_transition_soc - 1)) * (
            soc - 1
        )

    dsoc = curr_soc - soc
    return dsoc # this is the soc change

print(df_dis['Two Stage Model'].iloc[:10])
max_battery_capacity = 30 #8
ef = 0.97
threshold = 0.90
for [i, row] in df_dis.iterrows():
    if i > 0:
        df_dis['Linear Model'].at[i] = df_dis['Linear Model'].iloc[i-1] + \
            ((ef*df_dis['current_setpoint'].iloc[i]*df_dis['V_L3_kocos_200'].iloc[i]/1000* math.sqrt(3) * timestep/60) / max_battery_capacity) * 100
    
    if i > 0:        
        if df_dis['Two Stage Model'].iloc[i-1] < threshold*100:
            df_dis['Two Stage Model'].at[i] = df_dis['Two Stage Model'].iloc[i-1] +\
                (ef * df_dis['current_setpoint'].iloc[i-1] * df_dis['V_L3_kocos_200'].iloc[i-1] * math.sqrt(3) / 1000 *
                    timestep/60) / max_battery_capacity * 100
        else:
            df_dis['Two Stage Model'].at[i] = df_dis['Two Stage Model'].iloc[i-1] + ef * two_stage_model(
                                                                df_dis['current_setpoint'].iloc[i-1],
                                                                230, #df_dis['V_L3_kocos_200'].iloc[i-1],
                                                                # 230,
                                                                timestep,
                                                                max_battery_capacity,
                                                                threshold,
                                                                df_dis['Two Stage Model'].iloc[i-1]/100)*100

        if df_dis['Two Stage Model'].iloc[i] >= 100:
            df_dis['Two Stage Model'].at[i] = 100
print(df_dis['Two Stage Model'].iloc[:10])
# df_dis.plot(x="epoch", y=columns_to_plot_dis, subplots=True, figsize=(10, 10))
# plt.show()
# exit()

plt.figure(figsize=(10, 7))
# plt.style.use('seaborn-darkgrid')
# plt.style.use('seaborn')
# plt.rcParams.update({'font.size': 16})
plt.rcParams['font.family'] = ['serif']
plt.grid(True, which='major', axis='both')
plt.plot(df_dis.epoch.iloc[:-10], df_dis[columns_to_plot_dis].iloc[:-10], linewidth=2.5)


#add xtixks, yticks, grid, legend, title, and labels, and increase font size
plt.xticks(fontsize=28, rotation=45)
#change xtick labels to show only hours and minutes
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.yticks(fontsize=28)
# plt.legend(loc='center left', fontsize=28)
#write legend text
plt.legend(['Actual SoC', 'Linear Model','Two Stage Model'], fontsize=28, loc='upper left')


# plt.title('Mitsubishi Outlander PHEV Charging Curve', fontsize=34)
# plt.xlabel('Time', fontsize=38)
plt.ylabel('State of Charge (%)', fontsize=32)
# plt.grid(True, which='minor', axis='both')
plt.xlim([df_dis.epoch.iloc[500], df_dis.epoch.iloc[-500]])

ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(df_dis.epoch.iloc[:-10], df_dis['current_setpoint'].iloc[:-10], linewidth=2.5, color='lightcoral',label='Setpoint')
ax2.plot(df_dis.epoch.iloc[:-10], df_dis['Output_Current_ID2'].iloc[:-10], linewidth=2, color='red', linestyle='--',label='Actual')


plt.ylabel('Current (A)', fontsize=32,color='lightcoral')    
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.yticks(np.arange(0, 101, 25)
    ,fontsize=28,color='lightcoral')

plt.xlim([df_dis.epoch.iloc[500], df_dis.epoch.iloc[-500]])
#turn off grid
ax2.grid(False)
plt.legend(loc='center right', fontsize=28)
#move legend over the grid lines




plt.tight_layout()  
plt.show()