#write script to read in data and do some basic EDA for the elaad data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#read in data from xlsx file
df = pd.read_excel('.\data\elaadnl_open_ev_datasets.xlsx',
                    sheet_name='open_transactions')

print(df.head())

#create a new column for hour of arrival and departure
df['hour_of_arrival'] = df['UTCTransactionStart'].dt.hour
df['hour_of_departure'] = df['UTCTransactionStop'].dt.hour

#create a new column for day of arrival and departure
df['day_of_arrival'] = df['UTCTransactionStart'].dt.day
df['day_of_departure'] = df['UTCTransactionStop'].dt.day

#sample from  df['hour_of_arrival'] to get a sense of the distribution
print(f'Hour of arrival: {df["hour_of_arrival"].sample(10)}')
print(f' Charge time: {df["ChargeTime"].sample(10)}')

print(df.nunique())

#plot histogram of uniqe chargepoint_IDs of the sum of the charge time
df.groupby('ChargePoint').hist(bins=850, figsize=(10,8))
plt.show()


#check for outliers
print(df.describe())

#plot histograms
df.hist(bins=31, figsize=(10,8))
# plt.show()

exit()
df = pd.read_excel('.\data\elaadnl_open_ev_datasets.xlsx',
                    sheet_name='open_metervalues')

print(df.head())
print(df.info())

#check for missing values
print(df.isnull().sum())

#check for duplicates
print(df.duplicated().sum())

#check for outliers
print(df.describe())

#plot histograms
df.hist(bins=50, figsize=(10,8))
plt.show()