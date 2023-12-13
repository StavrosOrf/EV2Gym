#here we will plot the data


import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import pylab as pl

df_ch = pd.read_csv('ch_honda_5.csv', sep=',')

df_ch = df_ch.iloc[300:, :]
# sample every second row
# df_ch = df_ch.iloc[::120, :]

#turn epoch into datetime
df_ch.epoch = pd.to_datetime(df_ch.epoch, unit="ms")
print(df_ch.epoch)
df_ch.reset_index(inplace=True)
# print(df_ch.ConnectorType_ID2)
# print(df_ch.ConnectorType_ID3)
# print(df_ch.ConnectorType_ID4)

port = 2 #2 for CCS, 3 for CHAdeMO, 4 for Type 2


columns_to_plot = ["P_kocos_200",
                  #  "AC_Power_set_ID4",
                #    "T_kocos_200",
                  #  "Q_kocos_200",
                #    "E_kocos_200",
                f"SoC_ID{port}",
                f"Output_Power_ID{port}",
                f"Output_Current_ID{port}",
                f"Output_Voltage_ID{port}",
                
                # "Output_Power_ID",
                # "AC_Power_set_ID2",                
                # "SoC_ID3",
                # "Output_Power_ID3",
                # "Output_Current_ID3", 
                # "Output_Voltage_ID3",
               #  "current_setpoint",
               #  "model_soc",
               #  "model_soc_l"
                   ]


# df_ch.plot(x="epoch", y=columns_to_plot, subplots=True, figsize=(20, 20))
# plt.show()

power_setpoint = 10 #kW

initial_soc = df_ch[f"SoC_ID{port}"].iloc[0]

print(f"initial_soc: {initial_soc}, power_setpoint: {power_setpoint}")
#find timestep by epoch column
timestep = df_ch.epoch.iloc[1] - df_ch.epoch.iloc[0]
#turn timestep into minutes
timestep = timestep.total_seconds()/60
print(f'timestep: {timestep} minutes')

battery_capacity = 36 # 36 kWh

df_ch['model_soc'] =  0
df_ch['model_soc'].at[0] = initial_soc 



# from scipy import optimize

# def segments_fit(X, Y, count):
#     xmin = X.min()
#     xmax = X.max()

#     seg = np.full(count - 1, (xmax - xmin) / count)

#     px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
#     py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

#     def func(p):
#         seg = p[:count - 1]
#         py = p[count - 1:]
#         px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
#         return px, py

#     def err(p):
#         px, py = func(p)
#         Y2 = np.interp(X, px, py)
#         return np.mean((Y - Y2)**2)

#     r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
#     return func(r.x)


# #plot the soc curve
# X = df_ch[f"SoC_ID{port}"].shift(+1).values.reshape(-1, 1)[1:]
# y = df_ch[f"SoC_ID{port}"].values.reshape(-1, 1)[1:]

# px, py = segments_fit(X, y, 3)
# # print coefficients and equations
# for i in range(len(px)-1):
#     print(f"{py[i]}*(x-{px[i]}) + ", end="")
# print(py[-1])

# taus = [20,80]

for i in range(1,len(df_ch)):
    power_setpoint = 10 #df_ch["P_kocos_200"].iloc[i-1]
#    change = ( power_setpoint * timestep/60)/battery_capacity * 100 #*100 because we want to have it in %
#    print(df_ch['model_soc'].iloc[i-1], f' + {change}')

    x = df_ch['model_soc'].iloc[i-1]

    change = ( power_setpoint * timestep/60)/battery_capacity * 100 #*100 because we want to have it in %
    df_ch['model_soc'].at[i] = df_ch['model_soc'].iloc[i-1] + change
# 
# pl.plot(X, y, ".")
# pl.plot(px, py, "-or")

#plot soc as line
plt.plot(df_ch.epoch, df_ch[f"SoC_ID{port}"])

plt.plot(df_ch.epoch, df_ch['model_soc'])

plt.show()

exit()