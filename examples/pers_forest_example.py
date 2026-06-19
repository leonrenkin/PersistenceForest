# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd

# %%
# Import PersistenceForest class
# PersistenceForest contains methods for plotting barcodes, cycle representatives, and generalized landscapes.
# This is the central class of the persforest package.
from persforest import PersistenceForest


# %%
#generate example point cloud
rng = np.random.default_rng(35)
num_points=300
points = rng.uniform(low=0.0, high=2*np.pi, size=num_points)
points = np.sqrt(np.abs(np.cos(1.5*points))+.1)[:,None] * np.column_stack((np.cos(points), np.sin(points))) + rng.normal(scale=0.05, size=(num_points,2))

plt.figure(figsize=(6,6))
plt.scatter( points[:,0], points[:,1], s = 3)
plt.axis('equal')

# %%
# Compute PersistenceForest object from point cloud
pers_forest = PersistenceForest(points)

# %%
# Plot barcode in codimension 1 with bars colored according to the forest structure
pers_forest.plot_barcode(min_bar_length=0.01, coloring = "forest")

# %%
# Plot point cloud and active cycles at different filtration values
pers_forest.plot_at_filtration(0.2)
pers_forest.plot_at_filtration(0.6)
pers_forest.plot_at_filtration(0.7)


# %%
#This cell showcases cycle representative extraction and plotting

# Plot barcode and cycle representatives at relative position 0.1 in the barcode with arbitrarily colored bars
pers_forest.plot_barcode(coloring="bars", sort = "birth", min_bar_length=0.01)
pers_forest.plot_barcode_cycle_reps(relative_position=0.1, min_bar_length=0.05, figsize=(6,6), coloring="bars", linewidth_cycle=2)

# extract cycle representatives at relative position 0.1
# List[SignedChain], each SignedChain represents a cycle in the forest
cycle_reps = pers_forest.barcode_cycle_reps(relative_position=0.1, min_bar_length=0.05)

#transform cycle representatives from chain to list of vertex coordinates
#forgets edge information, only keeps vertex coordinates
cycle_reps_vertex_coords = [cycle.vertex_coordinates(point_cloud=pers_forest.point_cloud) for cycle in cycle_reps]

print('Example of vertex cycle representatives coordinates')
print(cycle_reps_vertex_coords)  #print first cycle representative as array of coordinates

# %%
# showcase of generalized landscape functionalities

#import cycle functionals which map a (signed) cycle representative to a real number
from persforest.cycle_rep_vectorisations import signed_chain_edge_length, constant_one_functional,signed_chain_excess_curvature

#compute generalized landscapes
# By default (cache=True), landscapes are saved on the PersistenceForest object with the given label.
# Landscapes can be plotted from the PersistenceForest object with the chosen label.
pers_forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_edge_length,
    max_k=6,
    num_grid_points=1000,
    label="length",
    cache=True,
    cache_functionals=True,
)

pers_forest.compute_generalized_landscape_family(
    cycle_func=constant_one_functional,
    max_k=6,
    num_grid_points=1000,
    label="1",
)

pers_forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_excess_curvature,
    max_k=6,
    num_grid_points=1000,
    label="excess curvature",
)

# Plot the different landscape families
pers_forest.plot_landscape_family(label='length', title = "Length Persistence Landscapes")
pers_forest.plot_landscape_family(label="1", title = "Regular Persistence Landscapes")
pers_forest.plot_landscape_family(label="excess curvature", title = "Excess Curvature Persistence Landscapes")

# %%
print(pers_forest.landscape_families['length'])
print(pers_forest.barcode_functionals['length'])

# %%
from persforest.cycle_rep_vectorisations import signed_chain_excess_connected_components, signed_chain_area, signed_chain_connected_components, signed_chain_connected_components, signed_chain_excess_connected_components

# New point cloud example
double_edge_cloud = np.loadtxt("../point_cloud_csvs/signed_chain_example.csv",  delimiter=",", skiprows=1) * 100
double_edge_forest = PersistenceForest( point_cloud=double_edge_cloud )

double_edge_forest.plot_at_filtration(15,style_2d={"show_orientation_arrows": True})


# %%
# Showcase of signed vs unsigned chains
double_edge_forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_connected_components,
    max_k=6,
    num_grid_points=1000,
    label="signed components",
)

double_edge_forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_excess_connected_components,
    max_k=6,
    num_grid_points=1000,
    label="signed excess components",)

double_edge_forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_connected_components,
    max_k=6,
    num_grid_points=1000,
    label="unsigned components",
    signed=False)    #signed is True by default, set signed=False for unsigned versions

double_edge_forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_excess_connected_components,
    max_k=6,
    num_grid_points=1000,
    label="unsigned excess components",
    signed=False)   

double_edge_forest.plot_landscape_comparison_between_functionals(labels=["signed components", "unsigned components"] )
double_edge_forest.plot_landscape_comparison_between_functionals(labels=["signed excess components", "unsigned excess components"] )




# %%
# Sample landscapes on a fixed grid to get NumPy arrays.
grid = np.linspace(0.0, 1.0, 64)
length_family = pers_forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_edge_length,
    max_k=3,
    x_grid=grid,
    label="length-features",
    cache=False,
)
values = length_family.evaluate_on_grid(grid, levels=3)

# %%
