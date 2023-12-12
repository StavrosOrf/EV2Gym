import matplotlib.pyplot as plt
import networkx as nx
import PIL

#print current directory
import os
print(os.getcwd())

# Image URLs for graph nodes
icons = {
    "cpo": "icons/algorithm.png",
    "transformer": "icons/transformer.png",
    "charger_1": "icons/charging-station_1_port.png",
    "charger_2": "icons/charging-station_2_ports.png",
    "charger_wallbox": "icons/charger_wallbox.png",
    "ev": "icons/ev.png"
}

# Load images
images = {k: PIL.Image.open(fname) for k, fname in icons.items()}

# Generate the computer network graph
G = nx.Graph()

# G.add_node("cpo", image=images["cpo"])
for i in range(1, 3):
    G.add_node(f"transformer_{i}", image=images["transformer"])
    for j in range(1, 2):
        G.add_node("charger_2_" + str(i) + "_" + str(j), image=images["charger_2"])
        G.add_node("charger_1_" + str(i) + "_" + str(j), image=images["charger_1"])
        G.add_node("ev_1_" + str(i) + "_" + str(j), image=images["ev"])
        G.add_node("ev_2_" + str(i) + "_" + str(j), image=images["ev"])
        
        

# G.add_edge("cpo", "transformer_1")
# G.add_edge("cpo", "transformer_2")
# G.add_edge("cpo", "transformer_3")
for u in range(1, 3):
    for v in range(1, 2):
        G.add_edge("transformer_" + str(u), "charger_1_" + str(u) + "_" + str(v))
        G.add_edge("transformer_" + str(u), "charger_2_" + str(u) + "_" + str(v))
        G.add_edge("ev_1_" + str(u)+ "_" + str(v), "charger_2_" + str(u) + "_" + str(v))
        G.add_edge("ev_2_" + str(u)+ "_" + str(v), "charger_2_" + str(u) + "_" + str(v))
        

# Get a reproducible layout and create figure
pos = nx.spring_layout(G, seed=42)
pos_attrs = {}
for node, coords in pos.items():
    pos_attrs[node] = (coords[0], coords[1] + 0.12)
    
plt.ion()
fig, ax = plt.subplots()

# Note: the min_source/target_margin kwargs only work with FancyArrowPatch objects.
# Force the use of FancyArrowPatch for edge drawing by setting `arrows=True`,
# but suppress arrowheads with `arrowstyle="-"`
nx.draw_networkx_edges(
    G,
    pos=pos,
    ax=ax,
    arrows=True,
    arrowstyle="-",
    min_source_margin=15,
    min_target_margin=15,
)

#### Draw ev charging current
#draw nx labels for each edge from ev to charger
edge_labels = {("ev_1_1_1", "charger_2_1_1"): "I=1.5A",
               ("ev_2_1_1", "charger_2_1_1"): "I=1.5A",
                ("ev_1_2_1", "charger_2_2_1"): "I=1.5A",
                ("ev_2_2_1", "charger_2_2_1"): "I=1.5A"}
                                        
edge_labels_ax = nx.draw_networkx_edge_labels(G,
                             pos=pos,
                             edge_labels=edge_labels,
                             ax=ax,
                             font_color="b",                            
                             )

#### Draw charger current
#draw nx labels for each edge from charger to transformer
edge_labels = {("charger_2_1_1", "transformer_1"): "I=3A",
               ("charger_2_2_1", "transformer_2"): "I=3A"}

nx.draw_networkx_edge_labels(G,
                            pos=pos,
                            edge_labels=edge_labels,
                            ax=ax,
                            font_color="r",                            
                            )

#### Draw loading of transformer
#draw node labels for each transformer
node_labels = {"transformer_1": "50%",
               "transformer_2": "50%"}

nx.draw_networkx_labels(G,
                        pos=pos_attrs,
                        labels=node_labels,
                        ax=ax,
                        verticalalignment = 'top',                        
                        font_color="r"                        
                        )

#### Draw ev SOC
# draw node labels for each ev
node_labels = {"ev_1_1_1": "90%",
               "ev_2_1_1": "40%",
                "ev_1_2_1": "80%",
                "ev_2_2_1": "40%"}

nx.draw_networkx_labels(G,
                        pos=pos_attrs,
                        labels=node_labels,
                        ax=ax,
                        verticalalignment = 'top',                        
                        font_color="g"                        
                        )
               

# Transform from data coordinates (scaled between xlim and ylim) to display coordinates
tr_figure = ax.transData.transform
# Transform from display to figure coordinates
tr_axes = fig.transFigure.inverted().transform

# Select the size of the image (relative to the X axis)
icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025
icon_center = icon_size / 2.0

# Add the respective image to each node
for n in G.nodes:
    xf, yf = tr_figure(pos[n])
    xa, ya = tr_axes((xf, yf))
    # get overlapped axes and plot icon
    a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size]) 
    print(type(a))   
    a.imshow(G.nodes[n]["image"])
    a.axis("off")
    
# add title to top of the figure 
# fig.suptitle("EV Charging Network", fontsize=16)

# add icon to the bottom of the figure
icon = PIL.Image.open("icons/logo.png")
icon = icon.resize((100, 50), PIL.Image.ANTIALIAS)
fig.figimage(icon, 20, 20, alpha=1, zorder=1)

# add text to a box in the bottom right corner
text = "Time step: 0 / 96 \n Time: 12:00 Date: 01.01.2021\nSimulation Name: Test"
text1 = fig.text(
    0.99,
    0.01,
    text,
    wrap=True,
    horizontalalignment="right",
    fontsize=12,
    bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
)
print(text1.get_text())
print(type(text1))

# plt.show()
print(edge_labels_ax)
import numpy as np
for i, phase in enumerate(np.linspace(0, 10*3.14, 500)):
    text1.set_text(f'text = "Time step: {i} / 96 \n Time: 12:00 Date: 01.01.2021\nSimulation Name: Test" {i}')
    edge_labels_ax[('ev_1_1_1', 'charger_2_1_1')].set_text(str(np.sin(phase)))
    fig.canvas.draw()
    fig.canvas.flush_events()
