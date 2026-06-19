from pathlib import Path
import numpy as np
import pandas as pd
import json

def generate_params(
    n: int,
    seed: int = 0,
    tendril_range=(0, 20),  # inclusive lower, inclusive upper
    tendril_length_range=(0.1, 0.4),
    mode_scenarios=(
        ("single", "uniform", 0.5),
        ("branching", "uniform", 0.25),
        ("single", "polarized", 0.25),
    ),
    branching_probability_range=(0.6, 0.8),
):
    """Generate valid parameter rows for parametric loop-tendril point clouds."""
    if n < 0:
        raise ValueError("n must be non-negative")

    tendril_low, tendril_high = map(int, tendril_range)
    if tendril_low < 0 or tendril_low > tendril_high:
        raise ValueError("tendril_range must be a non-negative inclusive range")

    length_low, length_high = map(float, tendril_length_range)
    if length_low < 0.0 or length_low > length_high:
        raise ValueError("tendril_length_range must be a non-negative range")
    if tendril_high > 0 and length_high <= 0.0:
        raise ValueError("tendril_length_range must include positive values when tendrils can be generated")

    branch_low, branch_high = map(float, branching_probability_range)
    if branch_low < 0.0 or branch_high > 1.0 or branch_low > branch_high:
        raise ValueError("branching_probability_range must lie within [0, 1]")
    if len(mode_scenarios) == 0:
        raise ValueError("mode_scenarios must contain at least one scenario")

    scenario_modes = []
    scenario_probabilities = []
    for scenario in mode_scenarios:
        try:
            tendril_mode, root_mode, probability = scenario
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "mode_scenarios entries must be (tendril_mode, root_mode, probability)"
            ) from exc
        scenario_modes.append((tendril_mode, root_mode))
        scenario_probabilities.append(float(probability))

    probabilities = np.asarray(scenario_probabilities, dtype=float)
    if np.any(probabilities < 0.0):
        raise ValueError("mode_scenarios probabilities must be non-negative")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("mode_scenarios probabilities must sum to 1.0")

    rng = np.random.default_rng(seed)
    n_tendrils = rng.integers(tendril_low, tendril_high + 1, size=n)
    tendril_lengths = rng.uniform(length_low, length_high, size=n)

    positive_length = max(length_low, np.nextafter(0.0, 1.0))
    tendril_lengths = np.where(n_tendrils == 0, 0.0, np.maximum(tendril_lengths, positive_length))

    sampled_scenarios = rng.choice(len(scenario_modes), size=n, p=probabilities)
    sampled_modes = np.asarray([scenario_modes[i][0] for i in sampled_scenarios])
    sampled_root_modes = np.asarray([scenario_modes[i][1] for i in sampled_scenarios])
    branching_probabilities = rng.uniform(branch_low, branch_high, size=n)
    branching_probabilities = np.where(
        (n_tendrils > 0) & (sampled_modes == "branching"),
        branching_probabilities,
        0.0,
    )

    params = pd.DataFrame({
        "sample_id": np.arange(n),
        "n_tendrils": n_tendrils,
        "tendril_length": tendril_lengths,
        "tendril_mode": sampled_modes,
        "root_mode": sampled_root_modes,
        "branching_probability": branching_probabilities,
    })
    return params


if __name__ == "__main__":

    out_dir = Path("classification_benchmarks/outputs")
    out_dir.mkdir(exist_ok=True)

    metadata = {
        "dataset": "loop_tendril_point_clouds",
        "n_samples": 1000,
        "parameter_seed": 123,
        "sampling_seed": 123,
        "death_normalization_target": 1.0,
        "parameter_generation": {
            "tendril_range": [0, 20],  # inclusive lower, inclusive upper
            "tendril_length_range": [0.1, 0.4],
            "mode_scenarios": [
                ["single", "uniform", 0.5],
                ["branching", "uniform", 0.25],
                ["single", "polarized", 0.25],
            ],
            "branching_probability_range": [0.5, 0.8],
        },
        "graph_generation": {
            "normalize": False,
            "n_loop_vertices": 80,
            "loop_noise": 0.01,
        },
        "scale_estimation_sampling": {
            "target_spacing": 0.15,
            "pre_sample_noise_std": 0.01,
            "post_sample_noise_std": 0.0008,
        },
        "final_sampling": {
            "target_spacing": 0.015,
            "pre_sample_noise_std": 0.0,
            "post_sample_noise_std": 0.0,
        },
        "outputs": {
            "params_csv": "loop_tendril_branched_polarized_params.csv",
            "metadata_json": "loop_tendril_branched_polarized_metadata.json",
            "point_cloud_dir": "point_clouds_branched_polarized",
        },
    }

    parameter_config = metadata["parameter_generation"]
    graph_config = metadata["graph_generation"]
    scale_sampling_config = metadata["scale_estimation_sampling"]
    final_sampling_config = metadata["final_sampling"]

    params = generate_params(
        n=metadata["n_samples"],
        seed=metadata["parameter_seed"],
        **parameter_config,
    )

    point_cloud_dir = out_dir / metadata["outputs"]["point_cloud_dir"]
    point_cloud_dir.mkdir(exist_ok = True)

    rows = []

    from loop_tendrils import generate_parametric_loop_tendrils
    from sampling import sample_planar_graph
    from loopforest import PersistenceForest

    for row in params.itertuples(index=False):
        graph = generate_parametric_loop_tendrils(
            n_tendrils=row.n_tendrils,          # type: ignore
            tendril_length=row.tendril_length,  # type: ignore
            tendril_mode=row.tendril_mode,      # type: ignore
            root_mode=row.root_mode,            # type: ignore
            branching_probability=row.branching_probability,# type: ignore
            **graph_config,
        )

        #determine scaling to rescale death to 1
        point_cloud = sample_planar_graph(
            graph=graph,
            seed=metadata["sampling_seed"],
            **scale_sampling_config,
        )
        forest = PersistenceForest(point_cloud)
        scaling = 1/forest.max_bar().death

        point_cloud = sample_planar_graph(
            graph=graph,
            target_spacing=final_sampling_config["target_spacing"] / scaling,
            pre_sample_noise_std=final_sampling_config["pre_sample_noise_std"] / scaling,
            post_sample_noise_std=final_sampling_config["post_sample_noise_std"] / scaling,
            seed=metadata["sampling_seed"],
            return_metadata=False,
        ) * scaling   # type: ignore

        save_path = point_cloud_dir / f"{row.sample_id}.npy"

        np.save(save_path, point_cloud)
        rows.append({
            "sample_id": row.sample_id,
            "n_points": len(point_cloud),
            "n_tendrils": row.n_tendrils,
            "tendril_length": row.tendril_length * scaling,  # type: ignore
            "param_tendril_length": row.tendril_length,
            "tendril_mode": row.tendril_mode,
            "root_mode": row.root_mode,
            "branching_probability": row.branching_probability,
            "scaling": scaling,
            "save_path": str(point_cloud_dir / f"{row.sample_id}.npy")
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / metadata["outputs"]["params_csv"], index=False)
    (out_dir / metadata["outputs"]["metadata_json"]).write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
