# This file implements functions that simulate the grid of the simulations using PandaPower
# NOT used yet, it is in development phase!!!!!!!!!!!!!!
from pandapower.plotting.plotly import pf_res_plotly
import pandapower as pp
import numpy as np
import time

from pandapower.plotting.plotly.mapbox_plot import set_mapbox_token
set_mapbox_token(
    'pk.eyJ1Ijoic29yZmFub3VkYWtpcyIsImEiOiJjbGs2ejQycGcwM3dwM21ydDd2dGc1dG1oIn0.dJ9E-1u53SZfDH0xgVaBkw')


class Grid:

    def __init__(self, case="kerber_landnetz_freileitung_1"):

        # Load the grid case
        if case == "kerber_dorfnetz":
            self.net = pp.networks.create_kerber_dorfnetz()
            load_bus_name = "loadbus"
            
        elif case == "kerber_landnetz_freileitung_2":
            self.net = pp.networks.create_kerber_landnetz_freileitung_2()
            load_bus_name = "bus"
                        
        else:
            raise NotImplementedError

        # add charging stations(storage units) and PV to every load (house) bus
        for i in range(len(self.net.bus)):
            # print(self.net.bus.loc[i])
            
            if load_bus_name in self.net.bus.at[i, 'name']:
                pp.create_storage(self.net,
                                  bus=i,
                                  p_mw=0,
                                  max_e_mwh=100,  # this parameter is not used in the PF calculation
                                  q_mvar=0,
                                  type="ChargingStation")

                pp.create_sgen(self.net,
                               bus=i,
                               p_mw=0,
                               q_mvar=0,
                               type="PV")

    def _get_grid_actions(self, actions):
        # Transform ations to the format (number_of_ports,1) -> (bus, p_mw)

        return actions

    def step(self, actions):

        actions = self._get_grid_actions(actions)

        # reset CS actions
        for i in range(len(self.net.storage)):
            self.net.storage.at[i, 'p_mw'] = 0

        # actions are in the format (bus, p_mw)
        assert (len(actions) == self.charging_stations)
        # for action in actions:
        #     self.net.storage.p_mw[action[0]] += action[1]

        std = 0.01
        # TODO: replace with REAL load profiles
        # self.net.load.p_mw = np.random.normal(
        #     self.net.load.p_mw, std, (len(self.net.load)))
        # self.net.load.q_mvar = np.random.normal(
        #     self.net.load.q_mvar, std, (len(self.net.load)))

        # # TODO: replace with REAL static generation (pv, wind) profiles
        # self.net.sgen.p_mw = np.random.normal(
        #     self.net.sgen.p_mw, std, (len(self.net.sgen)))
        # # self.net.sgen.q_mvar = np.random.normal(self.net.sgen.q_mvar, std, (len(self.net.sgen)))

        # # TODO: replace with REAL generation profiles
        # self.net.gen.p_mw = np.random.normal(
        #     self.net.gen.p_mw, std, (len(self.net.gen)))
        # self.net.gen.vm_pu = np.random.normal(
        #     self.net.gen.vm_pu, std, (len(self.net.gen)))

        # TODO: replace with REAL ext_grid profile
        # self.net.ext_grid.vm_pu = np.random.normal(
        #     self.net.ext_grid.vm_pu, std, (len(self.net.ext_grid)))
        # self.net.ext_grid.va_degree = np.random.normal(
        #     self.net.ext_grid.va_degree, std, (len(self.net.ext_grid)))

        # solve the power flow
        pp.runpp(self.net, numba=True)

    def get_grid_state(self):
        '''
        TODO: return the grid state
        '''
        pass

    def get_charging_stations_buses(self):
        return self.charging_stations_buses

    # TODO: this functino returns all the related transformers to the buses of the charging stations,
    #  it will help in the future to get the transformer loading objectives
    def get_bus_transformers(self):
        # return (n_charging_stations) with calues from the set {n_transformers}
        return None

    def visualize_pf(self):

        # visualize the grid using plotly
        # pp.plotting.plotly.simple_plotly(net)
        pf_res_plotly(self.net)
        # simple plot of net
        # pp.plotting.simple_plot(self.net, plot_loads=True, plot_sgens=True)

    def get_overloaded_lines(self):
        return self.net.res_line[self.net.res_line.loading_percent > 100]

    def get_overloaded_trafos(self, verbose=True):
        if verbose:
            print(
                f'Traffo loading percent:\n {self.net.res_trafo.loading_percent}')
        return self.net.res_trafo[self.net.res_trafo.loading_percent > 100]


if __name__ == "__main__":

    # this is how Grid2op does it https://github.com/rte-france/Grid2Op/blob/master/grid2op/Environment/baseEnv.py#L1093

    grid = Grid(case="kerber_landnetz_freileitung_2")

    timeslot_duration = 15  # minutes
    simulation_duration = 100  # * timeslot_duration minutes

    for simu_step in range(simulation_duration):
        print(f"Simulation step {simu_step}")
        actions = [(1, 0.005), (2, 0.001), (2, -0.002), (3, 0.001)]
        # actions = [(10,0),(1,0),(2,0),(3,0)]
        grid.step(actions)
        grid.visualize_pf()
        # time.sleep(2)
        input("Press Enter to continue...")

        # print information about the overaloading and the lines and the transformer from here
        # print(grid.get_overloaded_lines())
        grid.get_overloaded_trafos(verbose=False)

    # Get informatino about the overaloading and the lines and the transformer from here
    # print(net.res_line)
    # print(net.res_trafo)
    # print(net.res_trafo.loading_percent.round(3).astype(str))
    # # visualize the grid using plotly
    # pp.plotting.plotly.simple_plotly(net)

    # net = grid.net
    # print(
    #     f'Overloaded lines: {net.res_line[net.res_line.loading_percent > 100]}')
    # pp.plotting.to_html(net, "./test.html")

    # pf_res_plotly(net)

    # # simple plot of net
    # pp.plotting.simple_plot(net, plot_loads=True, plot_sgens=True)
