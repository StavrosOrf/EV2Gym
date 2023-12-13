from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
"this script reads ev charging and discharge curves from a csv file and plots them"

import pandas as pd
import matplotlib.pyplot as plt
import math
# read csv file
# df = pd.read_csv("./dis_1.csv", sep=",")
# df = pd.read_csv("./dis_honda_2.csv", sep=",")

# read csv and remove \" from every line

# open csv file and remove " from every line
# with open('ch_tesla_1.csv', 'r') as f:
#    lines = f.readlines()
#    with open('ch_tesla_2.csv', 'w') as f1:
#       for line in lines:
#          f1.write(line.replace('"', ''))


df = pd.read_csv('ch_tesla_2.csv', sep=',')
# df = pd.read_csv('ch_honda_5.csv', sep=',')

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

                   # "Output_Power_ID",                   
                   "V_L1_kocos_200",
                     "I_L1_kocos_200",                     
                   # "SoC_ID3",
                   # "Output_Power_ID3",
                   # "Output_Current_ID3",
                   # "Output_Voltage_ID3",
                  #  "current_setpoint",
                  #  "model_actual_curent",
                    "Linear Model",
                    "SoC"
                   ]

#plot the current
df["current_setpoint"] = 55.22
max_battery_capacity = 90 #57.5# kWh

df['power_setpoint'] = df['P_kocos_200'].iloc[20] * math.sqrt(3)
print(f'power_setpoint: {df["power_setpoint"].iloc[0]}')

# columns_to_plot_dis = ["Output_Current_ID2", "current_setpoint"]
columns_to_plot_ch = ["SoC", "Linear Model"]
df['SoC'] =  0
df['SoC'].at[0] = 6 #df_dis['SoC_ID2'].iloc[0]

df['Linear Model'] =  0 
df['Linear Model'].at[0] = df['SoC'].iloc[0]

df.epoch = pd.to_datetime(df.epoch, unit="ms")
df.reset_index(inplace=True)
timestep = df.epoch.iloc[1] - df.epoch.iloc[0]
#turn timestep into minutes
timestep = timestep.total_seconds()/60
print(f'timestep: {timestep} minutes')


ef = 0.8
for [i, row] in df.iterrows():
   if i > 0:
      df['SoC'].at[i] = df['SoC'].iloc[i-1] + (( ef * df['P_kocos_200'].iloc[i-1] * math.sqrt(3) * timestep/60) / max_battery_capacity) * 100
        
      if df['Linear Model'].iloc[i-1] >= 100:
         df['Linear Model'].at[i] = 100
      else:
         df['Linear Model'].at[i] = df['Linear Model'].iloc[i-1] + (( ef * df['power_setpoint'].iloc[i-1] * timestep/60) / max_battery_capacity) * 100

print(f'final SoC: {df["SoC"].iloc[-1]}')

df.plot(x="epoch", y=columns_to_plot_ch, subplots=False, figsize=(10, 10))
plt.show()

#
exit()

# create new column with current_setpoint
df["current_setpoint"] = df["P_kocos_200"] * \
    1000 / 394  # df[f"Output_Voltage_ID{port}"]
print(df[f"Output_Voltage_ID{port}"])
initial_soc = 8  # df[f"SoC_ID{port}"].iloc[0]
power_setpoint = df["P_kocos_200"].iloc[0]

print(f"initial_soc: {initial_soc}, power_setpoint: {power_setpoint}")
# find timestep by epoch column
timestep = df.epoch.iloc[1] - df.epoch.iloc[0]
# turn timestep into minutes
timestep = timestep.total_seconds()/60
print(f'timestep: {timestep} minutes')

battery_capacity = 28  # 36 kWh

df['model_soc'] = 0
df['model_soc'].at[0] = initial_soc
df['model_soc_l'] = 0
df['model_soc_l'].at[0] = initial_soc

# print the number of rows in df
print(len(df))
# for i in range(1,len(df)):
#    # print(f"current Energy transfered {((timestep/60)*power_setpoint)}")
#    df['model_soc'].iloc[i] = df['model_soc'].iloc[i-1] + \
#                         ((timestep/60)*power_setpoint)/(battery_capacity)


# fit a linear regression model to the data

data = df[["model_soc", f"SoC_ID{port}"]].dropna()
print(data.head())

# X is y shifted by 1 timestep
X = data[f"SoC_ID{port}"].shift(+1).values.reshape(-1, 1)[1:]
y = data[f"SoC_ID{port}"].values.reshape(-1, 1)[1:]

print(f'X[:10] {X[:10]}, y[:10] {y[:10]}')

reg = LinearRegression().fit(X, y)
print(reg.score(X, y))
print(reg.coef_)

# train a polynomial regression model

degree = 1
X_poly = PolynomialFeatures(degree).fit_transform(X)
reg2 = LinearRegression().fit(X_poly, y)
print(reg2.score(X_poly, y))
print(f'intercept: {reg2.intercept_}')
print(f'coefficients: {reg2.coef_}')


# write the linear regression equation
print(f"y = {reg.coef_[0][0]}x + {reg.intercept_[0]}")


# generate predictions
predictions = reg.predict(X)

# #plot the data
for i in range(1, len(df)):
    # print(f"current Energy transfered {((timestep/60)*power_setpoint)}")
    df['model_soc'].at[i] = reg2.predict(PolynomialFeatures(
        degree).fit_transform(df["model_soc"].iloc[i-1].reshape(-1, 1)))

    # if df['model_soc_l'].iloc[i-1] < 12:
    #    df['model_soc_l'].iloc[i] = df['model_soc_l'].iloc[i-1]
    # else:
    # print(df['model_soc_l'].iloc[i-1])
    df['model_soc_l'].at[i] = 1.0016586225495703 * \
        df['model_soc_l'].iloc[i-1] - 0.7006209393696281
    # + \
    #          ((timestep/60)*power_setpoint) /(battery_capacity)

    df['model_actual_curent'].at[i] = df["current_setpoint"].iloc[i] *\
        np.exp(100-df[f"SoC_ID{port}"].iloc[i])


# df[["model_soc", f"SoC_ID{port}"]].plot()
# plt.show()


# plot columns_to_plot from df
df[columns_to_plot].plot()


# show plot
plt.show()


# todo create a plot to compare
