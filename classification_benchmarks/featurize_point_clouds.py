from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from loopforest import PersistenceForest
from loopforest.cycle_rep_vectorisations import (
    constant_one_functional,
    signed_chain_avg_tendril_length,
    signed_chain_circularity_complement,
    signed_chain_connected_components_only_signed_simplices,
    signed_chain_excess_curvature,
    signed_chain_excess_curvature_diff_to_unsigned,
)

CycleFunc = Callable[..., float]


@dataclass(frozen=True)
class FunctionalConfig:
    label: str
    cycle_func: CycleFunc
    signed: bool


FUNCTIONAL_CONFIGS = (
    FunctionalConfig("standard_landscapes", constant_one_functional, False),
    FunctionalConfig("unsigned_excess_curvature", signed_chain_excess_curvature, False),
    FunctionalConfig("unsigned_circularity_complement", signed_chain_circularity_complement, False),
    FunctionalConfig("signed_excess_curvature", signed_chain_excess_curvature, True),
    FunctionalConfig("signed_circularity_complement", signed_chain_circularity_complement, True),
    FunctionalConfig(
        "excess_curvature_signed_unsigned_diff",
        signed_chain_excess_curvature_diff_to_unsigned,
        True,
    ),
    FunctionalConfig("avg_tendril_length", signed_chain_avg_tendril_length, True),
    FunctionalConfig("tendril_count", signed_chain_connected_components_only_signed_simplices, True),
)


def featurize_point_cloud(
    point_cloud,
    cycle_func,
    signed,
    max_k,
    x_grid,
    x_grid_first_layer=None,
    min_bar_length=0.0,
):
    forest = PersistenceForest(point_cloud)

    landscapes = forest.compute_generalized_landscape_family(
        cycle_func=cycle_func,
        label=cycle_func.__name__,
        x_grid=x_grid,
        signed=signed,
        max_k=max_k,
        min_bar_length=min_bar_length,
        cache=False,
    )

    if x_grid_first_layer is not None:
        landscape_first_layer = landscapes.evaluate_on_grid(
            grid=x_grid_first_layer,
            levels=1,
        ).flatten()
        landscape_higher_levels = landscapes.evaluate_on_grid(
            grid=x_grid,
            levels=range(2, max_k + 1),
        ).flatten()
        feature = np.concatenate((landscape_first_layer, landscape_higher_levels))
    else:
        feature = landscapes.evaluate_on_grid(grid=x_grid, levels=max_k).flatten()

    return feature


def process_cycle_func(
    output_dir,
    point_cloud_params_path,
    point_cloud_metadata_path,
    cycle_func,
    label,
    signed,
    landscape_feature_params,
):
    point_cloud_params = pd.read_csv(point_cloud_params_path)
    point_cloud_metadata = json.loads(
        Path(point_cloud_metadata_path).read_text(encoding="utf-8")
    )
    rows = []

    for point_cloud_param_row in point_cloud_params.to_dict(orient="records"):
        point_cloud = np.load(
            Path(__file__).resolve().parent / point_cloud_param_row["save_path"]
        )
        feature = featurize_point_cloud(
            point_cloud=point_cloud,
            cycle_func=cycle_func,
            signed=signed,
            **landscape_feature_params,
        )
        rows.append({
            **point_cloud_param_row,
            **{f"f{i}": value for i, value in enumerate(feature)},
        })

    feature_df = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / f"{label}.csv"
    metadata_path = output_dir / f"{label}_metadata.json"

    feature_df.to_csv(output_csv_path, index=False)

    landscape_params_metadata = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in landscape_feature_params.items()
    }

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "functional_label": label,
        "cycle_func": cycle_func.__name__,
        "signed": signed,
        "landscape_feature_params": landscape_params_metadata,
        "input_params_csv": str(point_cloud_params_path),
        "input_point_cloud_metadata_json": str(point_cloud_metadata_path),
        "output_csv": str(output_csv_path),
        "point_cloud_generation_metadata": point_cloud_metadata,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return feature_df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs" / "features"
    point_cloud_params_path = base_dir / "outputs" / "loop_tendril_params.csv"
    point_cloud_metadata_path = base_dir / "outputs" / "loop_tendril_metadata.json"

    landscape_feature_params = {
        "x_grid": np.linspace(0, 0.4, 21),
        "x_grid_first_layer": np.linspace(0, 1, 101),
        "max_k": 25,
        "min_bar_length": 0.0,
    }

    for config in FUNCTIONAL_CONFIGS:
        print(f"Starting {config.label}")
        process_cycle_func(
            output_dir=output_dir,
            point_cloud_params_path=point_cloud_params_path,
            point_cloud_metadata_path=point_cloud_metadata_path,
            cycle_func=config.cycle_func,
            label=config.label,
            signed=config.signed,
            landscape_feature_params=landscape_feature_params,
        )
