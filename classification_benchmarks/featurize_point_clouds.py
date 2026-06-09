from loopforest import PersistenceForest
import numpy as np
import pandas as pd
from pathlib import Path

from loopforest.cycle_rep_vectorisations import constant_one_functional, signed_chain_avg_tendril_length, signed_chain_excess_curvature, signed_chain_circularity_complement, signed_chain_excess_curvature_diff_to_unsigned, signed_chain_connected_components_only_signed_simplices

UNSIGNED_CYCLE_FUNCS = {"standard_landscapes": constant_one_functional,
                        "unsigned_excess_curvature": signed_chain_excess_curvature,
                        "unsigned_circularity_complement": signed_chain_circularity_complement}

SIGNED_CYCLE_FUNCS = {"signed_excess_curvature": signed_chain_excess_curvature,
                    "signed_circularity_complement": signed_chain_circularity_complement,
                    "excess_curvature_signed_unsigned_diff": signed_chain_excess_curvature_diff_to_unsigned,
                    "avg_tendril_length": signed_chain_avg_tendril_length,
                    "tendril_count": signed_chain_connected_components_only_signed_simplices}


def featurize_point_cloud(point_cloud, 
                          cycle_func, 
                          signed,
                          max_k, 
                          x_grid,  
                          x_grid_first_layer = None, 
                          min_bar_length = 0.0):
    
    forest = PersistenceForest(point_cloud)

    landscapes = forest.compute_generalized_landscape_family(cycle_func=cycle_func, 
                                                             label = cycle_func.__name__, 
                                                             x_grid=x_grid, 
                                                             signed=signed,
                                                             max_k = max_k,
                                                             min_bar_length = min_bar_length,
                                                             cache = False)

    if x_grid_first_layer is not None:
        landscape_first_layer = landscapes.evaluate_on_grid(grid=x_grid_first_layer, levels = 1)
        landscape_higher_levels = landscapes.evaluate_on_grid(grid=x_grid, levels = range(2,max_k+1)).flatten()
        feature = np.concatenate((landscape_first_layer,landscape_higher_levels))
    else:
        feature = landscapes.evaluate_on_grid(grid=x_grid, levels =max_k).flatten()

    return feature

def process_cycle_func(output_dir,
                       cycle_func, 
                       label,
                        signed,
                        landscape_feature_params):
    

    point_cloud_params = pd.read_csv("outputs/loop_tendril_params.csv")

    rows = []

    for row in point_cloud_params.itertuples(index=False):
        point_cloud = np.load(row.save_path) # type: ignore
        feature = featurize_point_cloud(point_cloud=point_cloud, cycle_func=cycle_func,signed=signed, **landscape_feature_params)
        rows.append({
            "sample_id": row.sample_id,
            **{f"f{i}": value for i, value in enumerate(feature)}
        })

    feature_df = pd.DataFrame(rows)
    Path(output_dir).mkdir(exist_ok = True)
    feature_df.to_csv(Path(output_dir) / f"{label}.csv", index=False)

    return feature_df



if __name__ == "main":

    output_dir = Path("outputs/features")
    Path(output_dir).mkdir(exist_ok = True)

    x_grid = np.linspace(0,0.4,20)
    x_grid_first_layer = np.linspace(0,0.4,20)

    label = "standard_landscapes"
    cycle_func =  constant_one_functional

    point_cloud_params = pd.read_csv("outputs/loop_tendril_params.csv")

    for name, cycle_func in UNSIGNED_CYCLE_FUNCS:
        
        print("To Do: implement")

    for name, cycle_func in SIGNED_CYCLE_FUNCS:
        
        print("To Do: implement")