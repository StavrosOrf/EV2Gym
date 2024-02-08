"""
This file is used to assess the battery drgradation model.

"""
import os
import sys
sys.path.append(os.path.realpath('../'))
from EVsSimulator.EVsSimulator.models.ev import EV

import numpy as np



def battery_degradation_test():
    '''
    This function print the battery degradation due to calendar aging and cycling 
    as a parameter of the charging power (x-axis) and the required energy per day (y-axis).
    '''

    timescale = 15  # 1 hour
    simulation_length = 96  # * 60  # 1 day
    # initialzing an EV

    amps_list = np.arange(8, 56, 1)
    required_energy_list = np.arange(5, 50, 1)

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
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(amps_list, required_energy_list)
    # ax.plot_surface(X, Y, calendar_degradation.T, label='calendar degradation')
    # ax.plot_surface(X, Y, cyclic_degradation.T, label='cycling degradation')
    # ax.set_xlabel('Charging power [A]')
    # ax.set_ylabel('Required energy per day [kWh]')
    # ax.set_zlabel('Battery degradation')
    # plt.show()

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


if __name__ == "__main__":
    battery_degradation_test()
