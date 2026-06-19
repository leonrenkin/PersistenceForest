from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from persforest import PersistenceForest
from persforest.cycle_rep_vectorisations import (
    constant_one_functional,
    signed_chain_avg_tendril_length,
    signed_chain_circularity_complement,
    signed_chain_connected_components_only_signed_simplices,
    signed_chain_excess_curvature,
    signed_chain_excess_curvature_diff_to_unsigned,
    signed_chain_tendril_branching_ratio
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
    #FunctionalConfig("excess_curvature_signed_unsigned_diff",signed_chain_excess_curvature_diff_to_unsigned,True,),
    FunctionalConfig("avg_tendril_length", signed_chain_avg_tendril_length, True),
    FunctionalConfig("tendril_count", signed_chain_connected_components_only_signed_simplices, True),
    #FunctionalConfig("tendril_branching_ratio", signed_chain_tendril_branching_ratio, True),
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


def _featurize_point_cloud_row(
    base_dir,
    row_index,
    point_cloud_param_row,
    cycle_func,
    signed,
    landscape_feature_params,
):
    point_cloud = np.load(point_cloud_param_row["save_path"])
    feature = featurize_point_cloud(
        point_cloud=point_cloud,
        cycle_func=cycle_func,
        signed=signed,
        **landscape_feature_params,
    )
    return row_index, {
        **point_cloud_param_row,
        **{f"f{i}": value for i, value in enumerate(feature)},
    }


def process_cycle_func(
    output_dir,
    point_cloud_params_path,
    point_cloud_metadata_path,
    cycle_func,
    label,
    signed,
    landscape_feature_params,
    progress_every=None,
    max_workers=1,
):
    point_cloud_params = pd.read_csv(point_cloud_params_path)
    point_cloud_metadata = json.loads(
        Path(point_cloud_metadata_path).read_text(encoding="utf-8")
    )
    if progress_every is not None and progress_every <= 0:
        raise ValueError("progress_every must be positive or None")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    base_dir = Path(__file__).resolve().parent
    point_cloud_param_rows = point_cloud_params.to_dict(orient="records")
    n_point_clouds = len(point_cloud_params)

    if max_workers == 1:
        rows = []
        for row_index, point_cloud_param_row in enumerate(point_cloud_param_rows):
            _, feature_row = _featurize_point_cloud_row(
                base_dir=base_dir,
                row_index=row_index,
                point_cloud_param_row=point_cloud_param_row,
                cycle_func=cycle_func,
                signed=signed,
                landscape_feature_params=landscape_feature_params,
            )
            rows.append(feature_row)
            completed = row_index + 1
            if progress_every is not None and completed % progress_every == 0:
                print(f"{label}: processed {completed}/{n_point_clouds} point clouds")
    else:
        indexed_rows = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _featurize_point_cloud_row,
                    base_dir,
                    row_index,
                    point_cloud_param_row,
                    cycle_func,
                    signed,
                    landscape_feature_params,
                )
                for row_index, point_cloud_param_row in enumerate(point_cloud_param_rows)
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                row_index, feature_row = future.result()
                indexed_rows[row_index] = feature_row
                if progress_every is not None and completed % progress_every == 0:
                    print(f"{label}: processed {completed}/{n_point_clouds} point clouds")
        rows = [
            indexed_rows[row_index]
            for row_index in range(len(point_cloud_param_rows))
        ]

    feature_df = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / f"{label}.csv"
    metadata_path = output_dir / f"{label}_metadata.json"

    feature_df.to_csv(output_csv_path, index=False)
    if progress_every is not None:
        print(f"{label}: wrote {output_csv_path}")

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
        "max_workers": max_workers,
        "output_csv": str(output_csv_path),
        "parallel": max_workers > 1,
        "point_cloud_generation_metadata": point_cloud_metadata,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return feature_df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs" / "features_single_uniform"
    point_cloud_params_path = base_dir / "outputs" / "loop_tendril_single_uniform_params.csv"
    point_cloud_metadata_path = base_dir / "outputs" / "loop_tendril_single_uniform_metadata.json"

    x_grid_first_layer = np.linspace(0, 1, 51)

    landscape_feature_params = {
        "x_grid": np.linspace(0, 0.4, 11),
        "x_grid_first_layer": x_grid_first_layer,
        "max_k": 20,
        "min_bar_length": 0.0,
    }
    max_workers = max((os.cpu_count() or 2) - 1, 1)

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
            progress_every=50,
            max_workers=max_workers,
        )
