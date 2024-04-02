# This file implements functions that simulate the grid of the simulations using PandaPower
#### NOT used yet, it is in development phase!!!!!!!!!!!!!!
from pandapower.plotting.plotly import pf_res_plotly
import pandapower as pp
import numpy as np

from pandapower.plotting.plotly.mapbox_plot import set_mapbox_token
set_mapbox_token(
    'pk.eyJ1Ijoic29yZmFub3VkYWtpcyIsImEiOiJjbGs2ejQycGcwM3dwM21ydDd2dGc1dG1oIn0.dJ9E-1u53SZfDH0xgVaBkw')


class Grid:

    def __init__(self, charging_stations, seed=9, case="default"):

        self.charging_stations = charging_stations
        # Load the grid case
        if case == "default":
            self.net = pp.networks.create_cigre_network_mv(with_der='pv_wind')
        else:
            raise NotImplementedError

        self.seed = seed
        np.random.seed(seed)

        # Initialize the Charging Stations
        for i in range(len(self.net.bus)):
            # Add a Storage Unit to each bus of the grid except the external grid bus
            if i not in self.net.ext_grid.bus.values:
                pp.create_storage(self.net, bus=i, p_mw=0, max_e_mwh=1000,
                                  q_mvar=0,  type="ChargingStation")
        # Initialize the charging stations conneted bus
        # get charging_stations random integers excluding the integers in self.net.ext_grid.bus.values
        self.charging_stations_buses = np.random.choice(
            [i for i in range(len(self.net.bus)) if i not in self.net.ext_grid.bus.values], charging_stations, replace=True)

    def _get_grid_actions(self,actions):
        # Transform ations to the format (number_of_ports,1) -> (bus, p_mw)

        return actions
    
    def step(self, actions):

        actions = _get_grid_actions(actions)

        # reset CS actions
        for i in range(len(self.net.storage)):
            self.net.storage.p_mw[i] = 0

        # actions are in the format (bus, p_mw)
        assert (len(actions) == self.charging_stations)
        for action in actions:
            self.net.storage.p_mw[action[0]] += action[1]

        std = 0.1
        # TODO: replace with REAL load profiles
        self.net.load.p_mw = np.random.normal(
            self.net.load.p_mw, std, (len(self.net.load)))
        self.net.load.q_mvar = np.random.normal(
            self.net.load.q_mvar, std, (len(self.net.load)))

        # TODO: replace with REAL static generation (pv, wind) profiles
        self.net.sgen.p_mw = np.random.normal(
            self.net.sgen.p_mw, std, (len(self.net.sgen)))
        # self.net.sgen.q_mvar = np.random.normal(self.net.sgen.q_mvar, std, (len(self.net.sgen)))

        # TODO: replace with REAL generation profiles
        self.net.gen.p_mw = np.random.normal(
            self.net.gen.p_mw, std, (len(self.net.gen)))
        self.net.gen.vm_pu = np.random.normal(
            self.net.gen.vm_pu, std, (len(self.net.gen)))

        # TODO: replace with REAL ext_grid profile
        self.net.ext_grid.vm_pu = np.random.normal(
            self.net.ext_grid.vm_pu, std, (len(self.net.ext_grid)))
        self.net.ext_grid.va_degree = np.random.normal(
            self.net.ext_grid.va_degree, std, (len(self.net.ext_grid)))

        # solve the power flow
        pp.runpp(self.net, numba=False)
    
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

    grid = Grid(4)

    timeslot_duration = 5  # minutes
    simulation_duration = 3  # * timeslot_duration minutes

    for simu_step in range(simulation_duration):
        print(f"Simulation step {simu_step}")
        actions = [(1, 0.5), (2, 0.1), (2, -0.2), (3, 0.1)]
        grid.step(actions)
        # grid.visualize_pf()

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
