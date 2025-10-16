'''this data contains the network files including nodes and lines data the format is
for nodes: we have 2 columns indicating that whether the node is slack bus or bus numbers
for lines: we have line_df = pd.DataFrame({
    'FROM',
    'TO',
    'R',
    'X',
    'B',
    'STATUS',
    'TAP': 1  # Assuming TAP is always 1
})'''
# the data sources: for 25, 69 and 123 nodes: https://github.com/DESL-EPFL/IEEE-benchmark-distribution-grids
