"this script reads ev charging and discharge curves from a csv file and plots them"

import pandas as pd
import matplotlib.pyplot as plt
#read csv file
df = pd.read_csv("./dis_1.csv", sep=",")
df = pd.read_csv("./dis_3_chademo_mitsubishi.csv", sep=",")
df = pd.read_csv("./unibid_1.csv", sep=",")
df = pd.read_csv("./bidir_1.csv", sep=",")

print(df.head())
print(df.columns)

#turn epoch into datetime
df.epoch = pd.to_datetime(df.epoch, unit="ms")
print(df.epoch)
print(df.ConnectorType_ID2)
print(df.ConnectorType_ID3)
print(df.ConnectorType_ID4)

port = 3 #2 for CCS, 3 for CHAdeMO

columns_to_plot = ["P_kocos_200",
                #    "T_kocos_200",
                   "Q_kocos_200",
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
                
                   ]
#plot columns_to_plot from df
df[columns_to_plot].plot()


#show plot
plt.show()
