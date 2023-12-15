from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
"this script reads ev charging and discharge curves from a csv file and plots them"

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.dates as mdates
# read csv file
# df = pd.read_csv("./dis_1.csv", sep=",")
# df = pd.read_csv("./dis_honda_2.csv", sep=",")

# read csv and remove \" from every line

# # open csv file and remove " from every line
# with open('ch_hyundai_5.csv', 'r') as f:
#    lines = f.readlines()
#    with open('ch_hyundai_6.csv', 'w') as f1:
#       for line in lines:
#          f1.write(line.replace('"', ''))


# df = pd.read_csv('ch_tesla_2.csv', sep=',')
# df = pd.read_csv('ch_honda_5.csv', sep=',')
df = pd.read_csv('ch_mitsu_1.csv', sep=',')


# df = pd.read_csv("./dis_3_chademo_mitsubishi.csv", sep=",")
# df = pd.read_csv("./unibid_1.csv", sep=",")
# df = pd.read_csv("./bidir_1.csv", sep=",")
# df = pd.read_csv("./ch_tesla_1.csv", sep=",")

print(df.head())
print(df.columns)

# df = df.iloc[300:, :]
# df = df.iloc[:15000, :]
# sample every second row
df = df.iloc[::120, :]

#extend df by 30 cells
for i in range(20):
    df = df._append(df.iloc[-1:, :])
    #fix epoch
    df.epoch.iloc[-1] = df.epoch.iloc[-2] + 1000*60


# turn epoch into datetime
df.epoch = pd.to_datetime(df.epoch, unit="ms")
print(df.epoch)
df.reset_index(inplace=True)
# print(df.ConnectorType_ID2)
# print(df.ConnectorType_ID3)
# print(df.ConnectorType_ID4)

port = 4  # 2 for CCS, 3 for CHAdeMO, 4 for Type 2


columns_to_plot = ["P_kocos_200",
                   #  "AC_Power_set_ID4",
                   #    "T_kocos_200",
                   #  "Q_kocos_200",
                   #    "E_kocos_200",
                   #  f"SoC_ID{port}",
                   f"Output_Power_ID{port}",
                   f"Output_Current_ID{port}",
                   f"Output_Voltage_ID{port}",
                   "AC_Power_set_ID4",
                   # "Output_Power_ID",
                   "V_L1_kocos_200",
                   "I_L1_kocos_200",
                   # "SoC_ID3",
                   # "Output_Power_ID3",
                   # "Output_Current_ID3",
                   # "Output_Voltage_ID3",
                   #  "current_setpoint",
                   #  "model_actual_curent",
                   # "Linear Model",
                   # "SoC"
                   ]

# plot the current
df["current_setpoint"] = 55.22
max_battery_capacity = 20  # 57.5# kWh

df['power_setpoint'] = df['P_kocos_200'].iloc[20] * math.sqrt(3)
print(f'power_setpoint: {df["power_setpoint"].iloc[0]}')

# columns_to_plot_dis = ["Output_Current_ID2", "current_setpoint"]
columns_to_plot_ch = ["SoC", "Linear Model", 'Two Stage Model']
df['SoC'] = 0
df['SoC'].at[0] = 21  # df_dis['SoC_ID2'].iloc[0]

df['Linear Model'] = 0
df['Linear Model'].at[0] = df['SoC'].iloc[0]

df['Two Stage Model'] = 0
df['Two Stage Model'].at[0] = df['SoC'].iloc[0]

df.epoch = pd.to_datetime(df.epoch, unit="ms")
df.reset_index(inplace=True)
timestep = df.epoch.iloc[1] - df.epoch.iloc[0]
# turn timestep into minutes
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
        
    # self.current_capacity = curr_soc * max_battery_capacity

    # # For charging power and charging rate (current), we use the
    # # the average over this time period.
    # self.current_power = dsoc * max_battery_capacity  # / (period / 60)
    # self.required_power = self.required_power - self.current_power  # * period / 60
    # return self.current_power / (period / 60) * 1000 / voltage


ef = 0.76
threshold = 0.92
for [i, row] in df.iterrows():
    if i > 0:
     #   df['SoC'].at[i] = df['SoC'].iloc[i-1] + (( ef * df['P_kocos_200'].iloc[i-1] * math.sqrt(3) * timestep/60) / max_battery_capacity) * 100
        df['SoC'].at[i] = df['SoC'].iloc[i-1] + ((ef * df['Output_Current_ID4'].iloc[i-1] *
                                                 df['V_L3_kocos_200'].iloc[i-1] * math.sqrt(3) / 1000 * timestep/60) / max_battery_capacity) * 100

        if df['SoC'].iloc[i] >= 100:
                df['SoC'].at[i] = 100
        # if i > 230:
        #     power = df['Output_Current_ID4'].iloc[20] * 400 /1000
        # else:
        # power = df['Output_Current_ID4'].iloc[i-1] * 230 * math.sqrt(3) /1000
        power = 14.5 * df['V_L3_kocos_200'].iloc[i-1] * math.sqrt(3) / 1000
        if df['Output_Current_ID4'].iloc[i-1] > 1.4:
            df['Linear Model'].at[i] = df['Linear Model'].iloc[i-1] + \
                ((ef * power *
                  timestep/60) / max_battery_capacity) * 100

            if df['Linear Model'].iloc[i] >= 100:
                df['Linear Model'].at[i] = 100

        else:
            df['Linear Model'].at[i] = df['Linear Model'].iloc[i-1] +\
                (ef * df['Output_Current_ID4'].iloc[i-1] * df['V_L3_kocos_200'].iloc[i-1] * math.sqrt(3) / 1000 *
                    timestep/60) / max_battery_capacity * 100

        if df['Two Stage Model'].iloc[i-1] < threshold*100:
            df['Two Stage Model'].iloc[i] = df['Two Stage Model'].iloc[i-1] +\
                (ef * df['Output_Current_ID4'].iloc[i-1] * df['V_L3_kocos_200'].iloc[i-1] * math.sqrt(3) / 1000 *
                 timestep/60) / max_battery_capacity * 100
        else:
            df['Two Stage Model'].at[i] = df['Two Stage Model'].iloc[i-1] + ef * two_stage_model(
                                                                14.5,
                                                                # df['V_L3_kocos_200'].iloc[i-1],
                                                                230,
                                                                timestep,
                                                                max_battery_capacity,
                                                                threshold,
                                                                df['Two Stage Model'].iloc[i-1]/100)*100

            if df['Two Stage Model'].iloc[i] >= 100:
                df['Two Stage Model'].at[i] = 100
                
print(f'final SoC: {df["SoC"].iloc[-1]}')

# df.plot(x="epoch", y=columns_to_plot, subplots=True, figsize=(10, 10))
# plt.show()
# plt.exit()


plt.figure(figsize=(10, 7))
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 16})
plt.rcParams['font.family'] = ['serif']

plt.plot(df.epoch.iloc[20:], df[columns_to_plot_ch].iloc[20:], linewidth=2)

plt.legend(['Actual SoC', 'Linear Model', 'Two Stage\n   Model'], fontsize=28,loc='upper left')
plt.ylabel('State of Charge (%)', fontsize=38)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.yticks(fontsize=28)
plt.xlabel('Time', fontsize=38)
# plt.legend(fontsize=28)
plt.grid(True, which='minor', axis='both')
plt.xlim(df.epoch.iloc[20], df.epoch.iloc[-1])
plt.xticks(fontsize=28, rotation=45)

#make new df column named "current setpoint" and fill it with 14.5 after index 180
df['current_setpoint'] = df['Output_Current_ID4']
df['current_setpoint'].iloc[180:] = 14.5

#plot the current setpoint and the actual current in separate axes on the same plot
#plot on the right y axis that is 30% of the height of the plot

ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(df.epoch.iloc[20:], df['current_setpoint'].iloc[20:], linewidth=2.5, color='lightcoral',label='Setpoint')
ax2.plot(df.epoch.iloc[20:], df['Output_Current_ID4'].iloc[20:], linewidth=2.5, color='red', linestyle='--',label='Actual')

plt.legend(loc='center right', fontsize=28)
plt.ylabel('Current (A)', fontsize=38,color='lightcoral')    
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.yticks(fontsize=28,color='lightcoral')
plt.ylim(0, 45)
#turn off grid
ax2.grid(False)

# plt.plot(df.epoch.iloc[20:], df['current_setpoint'].iloc[20:], linewidth=1.5)
# plt.plot(df.epoch.iloc[20:], df['Output_Current_ID4'].iloc[20:], linewidth=1.5)


# #make a magnified inset on the last 100 points of the curve
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
# import matplotlib.patches as patches
# import matplotlib.transforms as transforms

# axins = zoomed_inset_axes(plt.gca(), 2.5, loc='center right') # zoom-factor: 2.5, location: upper-left
# axins.plot(df.epoch.iloc[-100:], df[columns_to_plot_ch].iloc[-100:], linewidth=1.5)
# axins.set_xlim(df.epoch.iloc[-100], df.epoch.iloc[-1])
# axins.set_ylim(92, 100.1)
# plt.xticks(visible=False)
# plt.yticks(visible=False)
# # draw a bbox of the region of the inset axes in the parent axes and
# # connecting lines between the bbox and the inset axes area
# # mark_inset(plt.gca(), axins, loc1=2, loc2=4, fc="none", ec="0.5")
# print(df.epoch.iloc[-1]-df.epoch.iloc[-100])
# #add a rectangle to show the region of the inset axes
# rect = patches.Rectangle((df.epoch.iloc[-100],92),
#                          df.epoch.iloc[-1]-df.epoch.iloc[-100],
#                          1,
#                          linewidth=1,edgecolor='r',facecolor='none')
# plt.gca().add_patch(rect)

# #add a line to show the end of the curve
# plt.axvline(x=df.epoch.iloc[-1], color='r', linestyle='--')

# #add a line to show the start of the curve
# plt.axvline(x=df.epoch.iloc[20], color='r', linestyle='--')   

# #add xtixks, yticks, grid, legend, title, and labels, and increase font size
# plt.xticks(fontsize=28)
# #change xtick labels to show only hours and minutes
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


plt.tight_layout()  
plt.show()
