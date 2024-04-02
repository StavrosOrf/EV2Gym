import matplotlib.pyplot as plt
import networkx as nx
import PIL
import numpy as np
import pkg_resources

# Image URLs for graph nodes

icons = {
    "cpo": pkg_resources.resource_filename('ev2gym', "visuals/icons/cpo.png"),
    "transformer": pkg_resources.resource_filename('ev2gym', "visuals/icons/transformer.png"),
    "charger_1": pkg_resources.resource_filename('ev2gym', "visuals/icons/charging-station_1_port.png"),
    "charger_2": pkg_resources.resource_filename('ev2gym', "visuals/icons/charging-station_2_ports.png"),
    "charger_wallbox": pkg_resources.resource_filename('ev2gym', "visuals/icons/charger_wallbox.png"),
    "ev": pkg_resources.resource_filename('ev2gym', "visuals/icons/ev.png")
}


class Renderer():
    """Class for rendering the simulation environment"""

    def __init__(self, env):
        self.env = env

        # create networkx graph from environment
        self.G = nx.Graph()

        # Load images
        images = {k: PIL.Image.open(fname) for k, fname in icons.items()}

        # add nodes
        self.G.add_node("cpo", image=images["cpo"])

        for i in range(0, self.env.number_of_transformers):
            self.G.add_node(f"transformer_{i}", image=images["transformer"])
            # add edge to cpo
            self.G.add_edge("cpo", "transformer_" + str(i))

        for i in range(0, self.env.cs):
            if self.env.charging_stations[i].n_ports == 1:
                self.G.add_node("charger_" + str(i), image=images["charger_1"])
            elif self.env.charging_stations[i].n_ports == 2:
                self.G.add_node("charger_" + str(i), image=images["charger_2"])
            else:
                self.G.add_node("charger_" + str(i), image=images["charger_2"])

            # add edge to transformer
            self.G.add_edge(
                "transformer_" + str(self.env.charging_stations[i].connected_transformer), "charger_" + str(i))

            for j in range(0, self.env.charging_stations[i].n_ports):
                # add ev node
                self.G.add_node("ev_" + str(i) + "_" +
                                str(j), image=images["ev"])
                # add edge to charger
                self.G.add_edge("charger_" + str(i), "ev_" +
                                str(i) + "_" + str(j))

        # Get a reproducible layout and create figure
        pos = nx.spring_layout(self.G, seed=42)
        pos = nx.kamada_kawai_layout(self.G, pos=pos, weight=None)
        pos_attrs = {}
        for node, coords in pos.items():
            pos_attrs[node] = (coords[0], coords[1] + 0.14)

        pos__down_attrs = {}
        for node, coords in pos.items():
            pos__down_attrs[node] = (coords[0], coords[1] - 0.14)

        plt.ion()
        self.fig, ax = plt.subplots()

        # Note: the min_source/target_margin kwargs only work with FancyArrowPatch objects.
        # Force the use of FancyArrowPatch for edge drawing by setting `arrows=True`,
        # but suppress arrowheads with `arrowstyle="-"`
        self.graph_edges = nx.draw_networkx_edges(
            self.G,
            pos=pos,
            ax=ax,
            arrows=True,
            arrowstyle="-",
            min_source_margin=15,
            min_target_margin=15,
        )
    
        #change the linestype of the edges connectign transformer to cpo
        for i in range(0, self.env.number_of_transformers):
            self.graph_edges[i].set_linestyle("--")
            self.graph_edges[i].set_linewidth(3)
            

        # Drawnx labels for each charging station node to show power

        node_labels = {}
        for i in range(0, self.env.cs):
            node_labels["charger_" +
                        str(i)] = str(round(self.env.charging_stations[i].current_power_output, 1)) + "kW"

        self.charger_labels_ax = nx.draw_networkx_labels(self.G,
                                                         pos=pos__down_attrs,
                                                         labels=node_labels,
                                                         ax=ax,
                                                         verticalalignment='bottom',
                                                         font_size=8,
                                                         font_color="blue"
                                                         )

        # Draw charger current
        # draw nx labels for each edge from charger to transformer

        edge_labels = {}
        for i in range(0, self.env.cs):
            for j in range(0, self.env.charging_stations[i].n_ports):
                edge_labels[("charger_" + str(i), "transformer_" + str(self.env.charging_stations[i].connected_transformer))] = str(
                    round(self.env.charging_stations[i].current_total_amps, 1)) + "A"

        self.charger_to_tr_axes = nx.draw_networkx_edge_labels(self.G,
                                                               pos=pos,
                                                               edge_labels=edge_labels,
                                                               ax=ax,
                                                               font_color="r",
                                                               )

        # Draw node label cpo total power consumed

        self.cpo_power_ax = nx.draw_networkx_labels(self.G,
                                                    pos=pos__down_attrs,
                                                    labels={"cpo": str(
                                                    round(self.env.current_power_usage[self.env.current_step], 1)) + "kW"},
                                                    ax=ax,
                                                    verticalalignment='bottom',
                                                    font_size=12,
                                                    font_color="blue"
                                                    )

        # Draw consumed power for each transformer
        # draw node labels for each transformer
        node_labels = {}
        for i in range(0, self.env.number_of_transformers):
            # append transformer current to node labels
            node_labels["transformer_" +
                        str(i)] = str(round(self.env.transformers[i].current_amps, 2)) + "A"

        self.tr_labels_ax = nx.draw_networkx_labels(self.G,
                                                    pos=pos_attrs,
                                                    labels=node_labels,
                                                    ax=ax,
                                                    verticalalignment='top',
                                                    font_color="orange"
                                                    )

        # draw edge labels for each ev to charger
        edge_labels = {}
        for i in range(0, self.env.cs):
            for j in range(0, self.env.charging_stations[i].n_ports):
                if self.env.charging_stations[i].evs_connected[j] != None:
                    edge_labels[("ev_" + str(i) + "_" + str(j), "charger_" + str(i))] = str(
                        round(self.env.charging_stations[i].evs_connected[j].actual_current, 1)) + "A"
                else:
                    edge_labels[("ev_" + str(i) + "_" + str(j),
                                 "charger_" + str(i))] = ""

        self.ev_to_charger_labels_ax = nx.draw_networkx_edge_labels(self.G,
                                                                    pos=pos,
                                                                    edge_labels=edge_labels,
                                                                    ax=ax,
                                                                    font_color="r",
                                                                    font_size=8,
                                                                    )

        # Add label to the cpo node
        nx.draw_networkx_labels(self.G,
                                pos=pos_attrs,
                                labels={"cpo": "CPO"},
                                ax=ax,
                                verticalalignment='top',
                                font_color="black"
                                )

        # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
        tr_figure = ax.transData.transform
        # Transform from display to figure coordinates
        tr_axes = self.fig.transFigure.inverted().transform

        # Select the size of the image (relative to the X axis)
        icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025
        icon_center = icon_size / 2.0

        self.ev_list = []
        # Add the respective image to each node
        for n in self.G.nodes:
            xf, yf = tr_figure(pos[n])
            xa, ya = tr_axes((xf, yf))
            # get overlapped axes and plot icon
            a = plt.axes(
                [xa - icon_center, ya - icon_center, icon_size, icon_size])
            a.imshow(self.G.nodes[n]["image"])
            a.axis("off")

            if n.startswith("ev"):
                self.ev_list.append(a)

        # add title to top of the figure
        # self.fig.suptitle("EV Charging Network", fontsize=16)

        # add icon to the bottom of the figure
        image_path = pkg_resources.resource_filename(
            'ev2gym', "visuals/icons/logo.png")
        icon = PIL.Image.open(image_path)
        icon = icon.resize((100, 50), PIL.Image.LANCZOS)
        self.fig.figimage(icon, 20, 20, alpha=1, zorder=1)

        # add text to a box in the bottom right corner
        text = f"Time step: {self.env.current_step}/ {self.env.simulation_length}\n" + \
               f"{self.env.sim_date}\n" \
            f"Simulation Name: {self.env.sim_name}"

        self.text_box = self.fig.text(
            0.99,
            0.01,
            text,
            wrap=True,
            horizontalalignment="right",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
        )

        # make ev nodes invisible
        counter = 0
        for i in range(0, self.env.cs):
            for j in range(0, self.env.charging_stations[i].n_ports):

                if self.env.charging_stations[i].evs_connected[j] == None:
                    self.ev_list[counter].set_visible(False)

                counter += 1

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render(self):
        # input("Press Enter to continue...")
        text = f"Time step: {self.env.current_step} / {self.env.simulation_length}\n" + \
               f"{self.env.sim_date}\n" \
            f"Scenario: {self.env.scenario}\n" \
            f"Simulation Name: {self.env.sim_name}\n"

        self.text_box.set_text(text)
        self.cpo_power_ax["cpo"].set_text(str(
            round(self.env.current_power_usage[self.env.current_step-1], 1)) + "kW")

        counter = 0
        for i in range(0, self.env.cs):
            for j in range(0, self.env.charging_stations[i].n_ports):
                if self.env.charging_stations[i].evs_connected[j] == None:
                    self.ev_list[counter].set_visible(False)
                    self.ev_to_charger_labels_ax[(
                        "ev_" + str(i) + "_" + str(j), "charger_" + str(i))].set_text("")
                else:
                    self.ev_list[counter].set_visible(True)
                    # set text to show soc
                    self.ev_list[counter].set_title(
                        str(round(
                            self.env.charging_stations[i].evs_connected[j].get_soc(), 1)*100) + "%",
                        fontsize=8,
                        color="green",)

                    self.ev_to_charger_labels_ax[("ev_" + str(i) + "_" + str(j), "charger_" + str(i))].set_text(str(
                        round(self.env.charging_stations[i].evs_connected[j].actual_current, 1)) + "A")

                counter += 1

            self.charger_to_tr_axes[("charger_" + str(i), "transformer_" + str(self.env.charging_stations[i].connected_transformer))
                                    ].set_text(str(round(self.env.charging_stations[i].current_total_amps, 1)) + "A")

            self.charger_labels_ax["charger_" + str(i)].set_text(
                str(round(self.env.charging_stations[i].current_power_output, 1)) + "kW")

        for i in range(0, self.env.number_of_transformers):
            # append transformer current to node labels
            self.tr_labels_ax["transformer_" +
                              str(i)].set_text(str(round(self.env.transformers[i].current_amps /
                                                         self.env.transformers[i].max_current[self.env.current_step-1]
                                                         *100, 0)) + "%")
        # update ev nodes
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
