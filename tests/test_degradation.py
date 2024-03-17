"""
This file is used to assess the battery drgradation model.

"""
import os
import sys
sys.path.append(os.path.realpath('../'))
from ev2gym.models.ev import EV

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import math


def battery_degradation_test():
    '''
    This function print the battery degradation due to calendar aging and cycling 
    as a parameter of the charging power (x-axis) and the required energy per day (y-axis).
    '''

    timescale = 15  # 1 hour
    simulation_length = 96  # * 60  # 1 day
    # initialzing an EV

    amps_list = np.arange(8, 56, 1)
    amps_list = [56]
    required_energy_list = np.arange(0, 50, 1)

    calendar_degradation = np.zeros(
        (len(amps_list), len(required_energy_list)))
    cyclic_degradation = np.zeros((len(amps_list), len(required_energy_list)))

    for i, amps in enumerate(amps_list):

        for j, required_energy in enumerate(required_energy_list):

            ev = EV(id=0,
                    location=0,
                    battery_capacity_at_arrival=50-required_energy,
                    battery_capacity=50,
                    time_of_arrival=0,
                    time_of_departure=simulation_length-1,
                    transition_soc=0.9999,
                    timescale=timescale,
                    )

            for t in range(simulation_length):
                current_energy, actual_current = ev.step(amps=amps,
                                                         voltage=230,
                                                         phases=3,
                                                         type='AC')
                # if ev.get_soc() >= 1:                    
                #     break    

            # for t in range(simulation_length//3):
            #     current_energy, actual_current = ev.step(amps=-amps,
            #                                             voltage=230,
            #                                             phases=3,
            #                                             type='AC')
            #     if ev.get_soc() < 0.5:
            #         break

            # for t in range(simulation_length//3):
            #     current_energy, actual_current = ev.step(amps=amps,
            #                                             voltage=230,
            #                                             phases=3,
            #                                             type='AC')
                # print(f'Battery capacity: {ev.get_soc():.2f},' +
                #     f'Controller current: {amps:.2f},' +
                #     f' Actual: {actual_current:.2f} A | {current_energy:.2f} kW')
            # exit()
            d_cal, d_cyc = ev.get_battery_degradation()

            # print(f'degradation due to calendar aging: {d_cal} \n' +
            #     f'degradation due to cycling: {d_cyc}')
            calendar_degradation[i, j] = d_cal
            cyclic_degradation[i, j] = d_cyc

    print(cyclic_degradation)
    # TODO subplots one below the other,
    # y-axis show degradation in both cases
    # 1st plot: calendar degradation as a function of average_soc in a day (x_axis)
    # 2nd plot: cyclic degradation as a function of the amount of energy exchanged in a day (cyclic depth, x-axis)
    # The units of degradation are Fraction of lost battery pack capacity for BM
    # d_cal = Calendar aging factor in BM
    
    def calendar_degradation_calc(avg_soc):
        e0 = 7.543e6
        e1 = 23.75e6
        e2 = 6976

        b_age = 2*365  # days        

        # Age of the battery in days
        T_acc = b_age

        # Simulation time in days
        T_sim = (simulation_length)*timescale/ (60*24) # days

        theta = 298.15  # Kelvin
        k = 0.8263  # Volts
        v_min = 3.3324  # Volts
        
        v_avg = v_min + k * avg_soc

        # alpha(v_avg)
        alpha = (e0 * v_avg - e1) * math.exp(-e2 / theta)
        d_cal = alpha * 0.75 * T_sim / (T_acc)**0.25
        return d_cal
    
    avg_soc = np.arange(0, 1, 0.01)
    calendar_degradation = np.zeros((len(avg_soc)))
    
    for i, soc in enumerate(avg_soc):
        calendar_degradation[i] = calendar_degradation_calc(soc) * 1e4
    cyclic_degradation = cyclic_degradation * 1e4
    
    font_size = 25

    plt.rcParams.update({'font.size': font_size})
    plt.rcParams['font.family'] = ['serif']
    fig, ax1 = plt.subplots()
    
    plt.plot(avg_soc, calendar_degradation,color='#93003a')
    ax1.set_xlabel('Average SOC in a day',color='#93003a')
    ax1.set_xticks(np.arange(0, 1.1, 0.2),
                     [f'{i:.1f}' for i in np.arange(0, 1.1, 0.2)],
                     color='#93003a')
    ax1.set_ylabel('Capacity Loss Fraction ($10^{-4}$)',fontsize=font_size-4)
    # ax1.ticklabel_format(axis='y', style='sci',
    #                      scilimits=(0,0),
    #                      useMathText=True,
    #                      useOffset=True)
    
    # move scilimits to the left
    # ax1.yaxis.major.formatter._useMathText = True
    # ax1.set_ylim(0,2)
    
    
    ax1.legend(["Calendar deg."],loc='upper left',bbox_to_anchor=(0, 0.7),)
    # 2nd plot: cyclic degradation as a function of the amount of energy exchanged in a day (cyclic depth, x-axis)
    # use a separate x-axis for the cyclic depth at the top of the plot
    
    ax2 = ax1.twiny()    
    ax2.plot(required_energy_list, cyclic_degradation[0],color='#00429d')
    ax2.set_xlabel('Energy Exchanged in a day (kWh)',color='#00429d')
    ax2.set_xticks(np.arange(0, 51, 10),
                   np.arange(0, 51, 10),
                   color='#00429d')
    # ax2.ylabel('Capacity Loss Fraction',color='b')        

    ax2.legend(["Cyclic deg."],loc='upper left',)
    #set legend color
    for text in ax1.get_legend().get_texts():
        text.set_color("#93003a")
        
    for text in ax2.get_legend().get_texts():
        text.set_color("#00429d")
        
    ax1.set_xlim(0,1)
    ax2.set_xlim(0,50)
    #grid lines
    ax1.grid()
    ax2.grid()
    plt.show()
    
    
    exit()
    # normalize the degradation
    # calendar_degradation = calendar_degradation / np.max(calendar_degradation)
    # cyclic_degradation = cyclic_degradation / np.max(cyclic_degradation)
    
    #use the same scale for both calendar and cycling degradation
    
    # calendar_degradation = calendar_degradation / np.max(cyclic_degradation)
    # cyclic_degradation = cyclic_degradation / np.max(cyclic_degradation)
    
    # calendar_degradation *= 1000
    # cyclic_degradation *= 1000
    # calendar_degradation[0,0] = np.max(cyclic_degradation)
    print(np.max(calendar_degradation))
    print(np.max(cyclic_degradation))
    

    # plot a 3d graph of the battery degradation as a function of the charging power and the required energy per day

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(amps_list, required_energy_list)
    # ax.plot_surface(X, Y, calendar_degradation.T, label='calendar degradation')
    # ax.plot_surface(X, Y, cyclic_degradation.T, label='cycling degradation')
    # ax.set_xlabel('Charging power [A]')
    # ax.set_ylabel('Required energy per day [kWh]')
    # ax.set_zlabel('Battery degradation')
    # ax.legend()
    # plt.show()
    font_size = 25
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams['font.family'] = ['serif']
    # plto a cmap of the battery degradation as a function of the charging power and the required energy per day
    # use common scale for both calendar and cycling degradation
    # do subplots for calendar and cycling degradation
    fig, ax = plt.subplots(1, 2, figsize=(15,6))
    # c = ax[0].pcolormesh(amps_list, required_energy_list,
    #                     calendar_degradation.T, cmap='viridis')
    cmap = plt.cm.viridis
    # norm = plt.Normalize(vmin=0, vmax=np.max(cyclic_degradation))
    ax[0].pcolormesh(amps_list, required_energy_list,
                         calendar_degradation.T, cmap=cmap)    
    ax[0].set_xlabel('Charging power [A]')
    ax[0].set_ylabel('Required energy per day [kWh]')
    ax[0].set_title('Calendar degradation')
    
    # fig.colorbar(c, ax=ax[0], label='Calendar degradation')
    
    
    #cmap take values from 0 to max value (calendar_degradation) and map them to colors (viridis cmap)
    
    c = ax[1].pcolormesh(amps_list, required_energy_list,
                        cyclic_degradation.T, cmap='viridis')
    
    ax[1].set_xlabel('Charging power [A]')
    ax[1].set_ylabel('Required energy per day [kWh]')
    ax[1].set_title('Cycling degradation')
    fig.colorbar(c, ax=ax, label='Capacity Loss(mAh)')
        
    plt.show()
    
    # fig, ax = plt.subplots()
    # c = ax.pcolormesh(amps_list, required_energy_list,
    #                   calendar_degradation.T, cmap='viridis')
    # ax.set_xlabel('Charging power [A]')
    # ax.set_ylabel('Required energy per day [kWh]')
    # fig.colorbar(c, ax=ax, label='Calendar degradation')
    # plt.show()

    # fig, ax = plt.subplots()
    # c = ax.pcolormesh(amps_list, required_energy_list,
    #                   cyclic_degradation.T, cmap='viridis')
    
    # ax.set_xlabel('Charging power [A]')
    # ax.set_ylabel('Required energy per day [kWh]')
    # fig.colorbar(c, ax=ax, label='Cycling degradation')
    # plt.show()

def battery_degradation_equations():
    pass

if __name__ == "__main__":
    battery_degradation_test()
