"""
    File to create Network features for Parler network
"""

import networkx as nx
import pandas as pd

# Path to data
path_data = "..\\..\\..\\Data\\output\\features_network\\"

# Load nodelist
node_list = pd.read_csv(path_data + "user_list.csv")
# Load edgelist
edge_list = pd.read_csv(path_data + "interactions.csv")

# Undirected network
# Create Parler graph
G = nx.from_pandas_edgelist(edge_list, "creator_origin", "creator_reply", edge_attr=["weights"], create_using=nx.DiGraph)

# Add node attributes
node_attr = node_list.set_index("creator").to_dict("index")
nx.set_node_attributes(G, node_attr)

# Centrality
# In-degree
in_degree = pd.DataFrame(G.in_degree).rename(columns={0: "index", 1: "degree_in"})
# Out-degree
out_degree = pd.DataFrame(G.out_degree).rename(columns={0: "index", 1: "degree_out"})

# Eigenvector centrality
# Left (corresponds to in-edges)
degree_eigen_left = pd.DataFrame.from_dict(nx.eigenvector_centrality(G), orient="index").reset_index(level=0).rename(columns={0: "eigen_left"})

# Betweenness centrality
betweenness = pd.DataFrame.from_dict(nx.betweenness_centrality(G, k=1000), orient="index").reset_index(level=0).rename(columns={0: "betweenness"})

# Combine network features
dfs_features = [in_degree, out_degree, degree_eigen_left, betweenness]
dfs = [df.set_index("index") for df in dfs_features]
features_network = pd.concat(dfs, axis=1)
features_network.index.names = ["creator"]

features_network.to_csv("..\\..\\..\\Data\\output\\" + "features_network.csv")

