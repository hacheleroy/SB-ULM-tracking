#!/usr/bin/env python3
"""Train the public TSBM implementation on one matched ULM crossing target."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from sbulm.calibration import physical_crossing_config
from sbulm.data import DetectionSequence
from sbulm.metrics import posterior_reference_metrics
from sbulm.simulation import simulate_crossing
from sbulm.tsbm_adapter import ULMGaussianLocalizationCost
from sbulm.tsbm_reference import (
    canonicalize_continuous_paths,
    canonicalize_global_particle_labels,
    collapse_global_label_probabilities,
    collapse_global_label_symmetry,
    compare_association_probabilities,
    compare_tsbm_association_samples,
    endpoint_permutation_joint,
    endpoint_permutation_marginals,
    enumerate_tsbm_brownian_localization_posterior,
    finite_sample_baseline,
    nearest_detection_associations,
    observation_set_rms,
    rao_blackwellized_association_probabilities,
    sample_exact_tsbm_paths,
    sliced_wasserstein_distance,
)

UPSTREAM_COMMIT = "c9dfec715266a2fd45e1ab195eb3b388fa6e8ee8"
POSITION_SCALE_MM = 0.010
REFERENCE_SPEED_MM_S = 6.84
REFERENCE_FRAME_RATE_HZ = 1_000.0


@dataclass(frozen=True)
class Regime:
    name: str
    separation_um: float
    localization_std_um: float
    simulation_seed: int


REGIMES = {
    "easy": Regime("easy", 20.0, 2.5, 0),
    "ambiguous": Regime("ambiguous", 0.0, 5.0, 0),
    "high_noise": Regime("high_noise", 0.0, 20.0, 0),
}


class NullPlotter:
    """Suppress upstream task-specific plots while retaining trainer hooks."""

    plot_cfg = None
    make_gif = False

    def __call__(self, *_args: object, **_kwargs: object) -> dict[str, float]:
        return {}


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def verify_upstream_checkout(path: Path) -> str:
    if not (path / "bridge" / "trainer_tsbm.py").is_file():
        raise FileNotFoundError(f"not a TSBM checkout: {path}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"TSBM checkout is {commit}; expected pinned commit {UPSTREAM_COMMIT}"
        )
    return commit


def normalized_sequence(sequence: DetectionSequence) -> DetectionSequence:
    return DetectionSequence(
        times=np.linspace(0.0, 1.0, sequence.n_frames),
        detections=sequence.detections,
        reference_ids=sequence.reference_ids,
    )


def calibrated_brownian_diffusion(n_frames: int) -> float:
    """Match one-frame 2-D Brownian RMS displacement to the speed design scale."""

    step_mm = REFERENCE_SPEED_MM_S / REFERENCE_FRAME_RATE_HZ
    normalized_step = 1.0 / (n_frames - 1)
    return step_mm / np.sqrt(2.0 * normalized_step)


def boundary_rows(
    frame: np.ndarray,
    *,
    n_rows: int,
    position_scale: float,
    seed: int,
) -> np.ndarray:
    frame_permutations = tuple(permutations(range(frame.shape[0])))
    if n_rows % len(frame_permutations) != 0:
        raise ValueError("boundary dataset size must be divisible by N!")
    rows = np.stack(
        [frame[list(permutation)].reshape(-1) for permutation in frame_permutations]
    )
    rows = np.tile(rows, (n_rows // rows.shape[0], 1)) / position_scale
    return np.random.default_rng(seed).permutation(rows).astype(np.float32)


def build_upstream_config(
    tsbm_root: Path,
    run_dir: Path,
    *,
    dimension: int,
    diffusion_std_scaled: float,
    train_size: int,
    batch_size: int,
    pretrain_epochs: int,
    finetune_epochs: int,
    spline_first_iterations: int,
    spline_iterations: int,
    training_seed: int,
    use_control_variate: bool,
    num_s_per_t: int,
    hidden_dim: int,
    time_embed_dim: int,
    spline_n_mean: int,
    spline_n_std: int,
    spline_grid_size: int,
    spline_time_samples: int,
    spline_path_samples: int,
    pretrain_refresh_epochs: int | None,
    finetune_refresh_epochs: int,
    finetune_start_direction: str,
    pretrain_lr: float,
    finetune_lr: float,
    resume_score_checkpoint: Path | None,
    resume_control_variate_checkpoint: Path | None,
    resume_optimizer_checkpoint: Path | None,
) -> Any:
    hydra = importlib.import_module("hydra")
    omegaconf = importlib.import_module("omegaconf")
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str((tsbm_root / "conf").resolve())
    ):
        config = hydra.compose(
            config_name="config", overrides=["setting=discrete_multi_2d"]
        )
    with omegaconf.open_dict(config):
        config.name = run_dir.name
        config.seed = training_seed
        config.device = "cpu"
        config.num_workers = 0
        config.pin_memory = False
        config.LOGGER = "NONE"
        config.make_gif = False
        config.dim = dimension
        config.cost_name = "quadratic"
        config.gsb_method = "tsbm"
        config.resume_from_checkpoint = resume_score_checkpoint is not None
        if resume_score_checkpoint is not None:
            config.resume_at_stage = "finetuning"
            config.resume_at_sgd_step = 1
            config.score_checkpoint_finetuning = str(
                resume_score_checkpoint.resolve()
            )
            config.cv_checkpoint_finetuning = (
                None
                if resume_control_variate_checkpoint is None
                else str(resume_control_variate_checkpoint.resolve())
            )
            config.optimizer_checkpoint_finetuning = (
                None
                if resume_optimizer_checkpoint is None
                else str(resume_optimizer_checkpoint.resolve())
            )
        config.checkpoint_dir = str((run_dir / "checkpoints").resolve())
        config.use_bidir = True
        config.use_control_variate = use_control_variate
        config.use_state_cost_annealing = True
        config.factor_state_cost = 1.0
        config.max_epoch_annealing = max(1, pretrain_epochs // 2)
        config.T = 1.0
        config.sigma = float(diffusion_std_scaled)
        config.eps_train = 1e-3
        config.eps_test = 1e-3
        config.train_num_steps = 40
        config.test_num_steps = 80
        config.log_num_times = 81
        config.batch_size = batch_size
        config.test_batch_size = min(512, train_size)
        config.spline_batch_size = train_size
        config.plot_npar = min(128, train_size)
        config.plot_num_times = 9
        config.test_epoch_stride = 1_000_000
        config.log_epoch_stride = max(1, pretrain_epochs // 5)
        config.save_epoch_stride = 1_000_000
        config.num_s_per_t = num_s_per_t
        config.score_lr = 3e-4
        config.cv_lr = 1e-4
        config.model.hidden_dim = hidden_dim
        config.model.num_hidden_layers = 2
        config.model.time_embed_dim = time_embed_dim
        config.control_variate.time_embed_dim = time_embed_dim
        config.bidir.pretrain_epochs = pretrain_epochs
        config.bidir.finetune_epochs = finetune_epochs
        config.bidir.pretrain_refresh_every = (
            pretrain_epochs
            if pretrain_refresh_epochs is None
            else pretrain_refresh_epochs
        )
        config.bidir.finetune_refresh_every = finetune_refresh_epochs
        config.bidir.pretrain_lr = pretrain_lr
        config.bidir.finetune_lr = finetune_lr
        config.bidir.pretrain_cv_lr = 1e-4
        config.bidir.finetune_cv_lr = 1e-5
        config.bidir.finetune_start_direction = finetune_start_direction
        config.spline.N_mean = spline_n_mean
        config.spline.N_std = spline_n_std
        config.spline.n_grid = spline_grid_size
        config.spline.T = spline_time_samples
        config.spline.N = spline_path_samples
        config.spline.nitr_first_it = spline_first_iterations
        config.spline.nitr = spline_iterations
        config.spline.lr_mean = 0.05
        config.spline.lr_gamma = 0.05
    return config


def make_trainer_class(base_class: type[Any]) -> type[Any]:
    class ULMTSBM(base_class):
        def __init__(self, *args: object, state_cost: Any, **kwargs: object) -> None:
            self._ulm_state_cost = state_cost
            super().__init__(*args, **kwargs)

        def build_state_cost(self) -> None:
            self.V = self._ulm_state_cost.to(self.device)
            self.disc_cost_times = self.V.y_times
            self.K = self.V.K
            self.discrete_cfg = {
                "cost_times": self.disc_cost_times,
                "num_cost_times": self.K,
            }
            self.V.get_num_cost_times_per_t = self.get_num_cost_times_per_t

        def get_plotter(self) -> NullPlotter:
            return NullPlotter()

    return ULMTSBM


def empirical_probabilities(
    samples: np.ndarray, posterior_associations: np.ndarray
) -> np.ndarray:
    lookup = {
        association.tobytes(): index
        for index, association in enumerate(posterior_associations)
    }
    counts = np.zeros(posterior_associations.shape[0], dtype=float)
    for sample in samples.astype(np.int64, copy=False):
        index = lookup.get(sample.tobytes())
        if index is not None:
            counts[index] += 1.0
    return counts / samples.shape[0]


def sample_trained_model(
    trainer: Any,
    sequence: DetectionSequence,
    *,
    n_samples: int,
    batch_size: int,
    position_scale: float,
    sample_seed: int,
    sample_direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch = importlib.import_module("torch")
    if sample_direction not in {"f", "b"}:
        raise ValueError("sample_direction must be 'f' or 'b'")
    boundary_index = 0 if sample_direction == "f" else -1
    starts = boundary_rows(
        sequence.detections[boundary_index],
        n_rows=n_samples,
        position_scale=position_scale,
        seed=sample_seed,
    )
    id_to_log = torch.linspace(
        0, trainer.test_N, sequence.n_frames, device=trainer.device
    ).round().long()
    sample_net = trainer.get_sample_net(sample_direction)
    path_batches = []
    logged_times = None
    trainer.set_seed(sample_seed)
    for start in range(0, n_samples, batch_size):
        x_start = torch.from_numpy(starts[start : start + batch_size]).to(trainer.device)
        paths, _, times = trainer.langevin.sample_sde(
            x_start,
            sample_net,
            sample_direction,
            trainer.test_sde_ts,
            id_to_log,
        )
        path_batches.append(paths.detach().cpu().numpy())
        logged_times = times.detach().cpu().numpy()
    flattened = np.concatenate(path_batches, axis=0)
    if sample_direction == "b":
        flattened = np.flip(flattened, axis=1).copy()
        logged_times = np.flip(logged_times).copy()
    paths = flattened.reshape(
        n_samples,
        sequence.n_frames,
        sequence.cardinalities[0],
        sequence.dimension,
    )
    return paths * position_scale, starts * position_scale, logged_times


def run(args: argparse.Namespace) -> dict[str, Any]:
    tsbm_root = args.tsbm_root.resolve()
    upstream_commit = verify_upstream_checkout(tsbm_root)
    sys.path.insert(0, str(tsbm_root))
    torch = importlib.import_module("torch")
    accelerate = importlib.import_module("accelerate")
    trainer_module = importlib.import_module("bridge.trainer_tsbm")
    torch.set_num_threads(args.threads)

    regime = REGIMES[args.regime]
    config = physical_crossing_config(
        central_separation_um=regime.separation_um,
        localization_std_um=regime.localization_std_um,
    )
    _, physical_sequence = simulate_crossing(config, seed=regime.simulation_seed)
    sequence = normalized_sequence(physical_sequence)
    diffusion_std_mm = calibrated_brownian_diffusion(sequence.n_frames)
    exact = enumerate_tsbm_brownian_localization_posterior(
        sequence,
        localization_std=config.localization_std,
        diffusion_std=diffusion_std_mm,
    )

    train_initial = boundary_rows(
        sequence.detections[0],
        n_rows=args.train_size,
        position_scale=POSITION_SCALE_MM,
        seed=args.training_seed,
    )
    train_final = boundary_rows(
        sequence.detections[-1],
        n_rows=args.train_size,
        position_scale=POSITION_SCALE_MM,
        seed=args.training_seed + 1,
    )
    test_initial = boundary_rows(
        sequence.detections[0],
        n_rows=args.test_size,
        position_scale=POSITION_SCALE_MM,
        seed=args.training_seed + 2,
    )
    test_final = boundary_rows(
        sequence.detections[-1],
        n_rows=args.test_size,
        position_scale=POSITION_SCALE_MM,
        seed=args.training_seed + 3,
    )
    tensor_dataset = torch.utils.data.TensorDataset
    init_ds = tensor_dataset(torch.from_numpy(train_initial))
    final_ds = tensor_dataset(torch.from_numpy(train_final))
    test_init_ds = tensor_dataset(torch.from_numpy(test_initial))
    test_final_ds = tensor_dataset(torch.from_numpy(test_final))

    normalized_times = sequence.times
    state_cost = ULMGaussianLocalizationCost(
        np.stack(sequence.detections)[1:-1] / POSITION_SCALE_MM,
        localization_std=config.localization_std / POSITION_SCALE_MM,
        y_times=normalized_times[1:-1],
        t_final=1.0,
        attr=(diffusion_std_mm / POSITION_SCALE_MM) ** 2,
        include_normalization=False,
    )
    run_dir = args.output_dir.resolve() / f"tsbm_wp2_{regime.name}"
    upstream_config = build_upstream_config(
        tsbm_root,
        run_dir,
        dimension=sequence.cardinalities[0] * sequence.dimension,
        diffusion_std_scaled=diffusion_std_mm / POSITION_SCALE_MM,
        train_size=args.train_size,
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        spline_first_iterations=args.spline_first_iterations,
        spline_iterations=args.spline_iterations,
        training_seed=args.training_seed,
        use_control_variate=args.use_control_variate,
        num_s_per_t=args.num_s_per_t,
        hidden_dim=args.hidden_dim,
        time_embed_dim=args.time_embed_dim,
        spline_n_mean=args.spline_n_mean,
        spline_n_std=args.spline_n_std,
        spline_grid_size=args.spline_grid_size,
        spline_time_samples=args.spline_time_samples,
        spline_path_samples=args.spline_path_samples,
        pretrain_refresh_epochs=args.pretrain_refresh_epochs,
        finetune_refresh_epochs=args.finetune_refresh_epochs,
        finetune_start_direction=args.finetune_start_direction,
        pretrain_lr=args.pretrain_lr,
        finetune_lr=args.finetune_lr,
        resume_score_checkpoint=args.resume_score_checkpoint,
        resume_control_variate_checkpoint=args.resume_control_variate_checkpoint,
        resume_optimizer_checkpoint=args.resume_optimizer_checkpoint,
    )
    accelerator = accelerate.Accelerator(cpu=True, split_batches=True)
    trainer_class = make_trainer_class(trainer_module.TSBM_Spline)
    start_time = time.perf_counter()
    with working_directory(run_dir):
        trainer = trainer_class(
            init_ds,
            final_ds,
            test_init_ds,
            test_final_ds,
            (upstream_config.dim,),
            upstream_config,
            accelerator=accelerator,
            state_cost=state_cost,
        )
        if args.skip_training:
            if args.resume_score_checkpoint is None:
                raise ValueError("--skip-training requires --resume-score-checkpoint")
        else:
            trainer.train()
        training_seconds = time.perf_counter() - start_time
        sampled_paths, sampled_starts, logged_times = sample_trained_model(
            trainer,
            sequence,
            n_samples=args.n_samples,
            batch_size=args.sample_batch_size,
            position_scale=POSITION_SCALE_MM,
            sample_seed=args.sample_seed,
            sample_direction=args.sample_direction,
        )

    sampled_associations = nearest_detection_associations(sampled_paths, sequence)
    comparison = compare_tsbm_association_samples(sampled_associations, exact)
    canonical_exact = collapse_global_label_symmetry(exact)
    canonical_samples = canonicalize_global_particle_labels(sampled_associations)
    canonical_comparison = compare_tsbm_association_samples(
        canonical_samples, canonical_exact
    )
    rao_blackwellized = rao_blackwellized_association_probabilities(
        sampled_paths,
        sequence,
        exact.associations,
        localization_std=config.localization_std,
    )
    rao_blackwellized_comparison = compare_association_probabilities(
        rao_blackwellized,
        exact,
        n_path_samples=args.n_samples,
    )
    canonical_support, canonical_rao_blackwellized = (
        collapse_global_label_probabilities(exact.associations, rao_blackwellized)
    )
    if not np.array_equal(canonical_support, canonical_exact.associations):
        raise RuntimeError("canonical association supports are not aligned")
    canonical_rao_blackwellized_comparison = compare_association_probabilities(
        canonical_rao_blackwellized,
        canonical_exact,
        n_path_samples=args.n_samples,
    )
    labelled_baseline = finite_sample_baseline(
        exact,
        n_samples=args.n_samples,
        n_replicates=args.baseline_replicates,
        seed=args.sample_seed + 1,
    )
    canonical_baseline = finite_sample_baseline(
        canonical_exact,
        n_samples=args.n_samples,
        n_replicates=args.baseline_replicates,
        seed=args.sample_seed + 2,
    )
    exact_paths, exact_path_associations = sample_exact_tsbm_paths(
        sequence,
        exact,
        localization_std=config.localization_std,
        diffusion_std=diffusion_std_mm,
        n_samples=args.n_samples,
        seed=args.sample_seed + 3,
    )
    exact_paths_second, exact_path_associations_second = sample_exact_tsbm_paths(
        sequence,
        exact,
        localization_std=config.localization_std,
        diffusion_std=diffusion_std_mm,
        n_samples=args.n_samples,
        seed=args.sample_seed + 4,
    )
    canonical_sampled_paths = canonicalize_continuous_paths(
        sampled_paths, sampled_associations
    )
    canonical_exact_paths = canonicalize_continuous_paths(
        exact_paths, exact_path_associations
    )
    canonical_exact_paths_second = canonicalize_continuous_paths(
        exact_paths_second, exact_path_associations_second
    )
    flattened_sampled = canonical_sampled_paths.reshape(args.n_samples, -1)
    flattened_exact = canonical_exact_paths.reshape(args.n_samples, -1)
    flattened_exact_second = canonical_exact_paths_second.reshape(args.n_samples, -1)
    sliced_wasserstein_um = 1_000.0 * sliced_wasserstein_distance(
        flattened_sampled,
        flattened_exact,
        n_projections=args.wasserstein_projections,
        seed=args.sample_seed + 5,
    )
    sliced_wasserstein_baseline_um = 1_000.0 * sliced_wasserstein_distance(
        flattened_exact,
        flattened_exact_second,
        n_projections=args.wasserstein_projections,
        seed=args.sample_seed + 5,
    )
    sampled_residuals_um = 1_000.0 * observation_set_rms(sampled_paths, sequence)
    exact_residuals_um = 1_000.0 * observation_set_rms(exact_paths, sequence)

    def residual_summary(residuals: np.ndarray) -> dict[str, float]:
        interior = residuals[:, 1:-1].reshape(-1)
        return {
            "interior_median_um": float(np.median(interior)),
            "interior_q90_um": float(np.quantile(interior, 0.9)),
            "initial_median_um": float(np.median(residuals[:, 0])),
            "final_median_um": float(np.median(residuals[:, -1])),
            "final_q90_um": float(np.quantile(residuals[:, -1], 0.9)),
        }
    empirical = empirical_probabilities(sampled_associations, exact.associations)
    exact_initial, exact_final = endpoint_permutation_marginals(exact)
    exact_endpoint_joint = endpoint_permutation_joint(exact)
    sample_initial = np.asarray(
        [
            np.mean(np.all(sampled_associations[:, 0] == permutation, axis=1))
            for permutation in np.unique(exact.associations[:, 0], axis=0)
        ]
    )
    sample_final = np.asarray(
        [
            np.mean(np.all(sampled_associations[:, -1] == permutation, axis=1))
            for permutation in np.unique(exact.associations[:, -1], axis=0)
        ]
    )
    endpoint_permutations = np.unique(exact.associations[:, 0], axis=0)
    sample_endpoint_joint = np.asarray(
        [
            [
                np.mean(
                    np.all(sampled_associations[:, 0] == initial, axis=1)
                    & np.all(sampled_associations[:, -1] == final, axis=1)
                )
                for final in endpoint_permutations
            ]
            for initial in endpoint_permutations
        ]
    )
    order = np.argsort(exact.probabilities)[::-1]
    leading_modes = [
        {
            "rank": rank + 1,
            "association": exact.associations[index].tolist(),
            "exact_mass": float(exact.probabilities[index]),
            "sample_frequency": float(empirical[index]),
            "rao_blackwellized_mass": float(rao_blackwellized[index]),
        }
        for rank, index in enumerate(order[:10])
    ]
    reference_metrics = posterior_reference_metrics(sequence, exact)
    summary: dict[str, Any] = {
        "experiment": "tsbm_wp2_matched_brownian_reference",
        "regime": asdict(regime),
        "upstream": {
            "repository": "https://github.com/maxencenoble/twisted-sb-matching",
            "commit": upstream_commit,
            "license": "MIT",
        },
        "physical_design": {
            "frame_rate_hz": REFERENCE_FRAME_RATE_HZ,
            "speed_mm_s": REFERENCE_SPEED_MM_S,
            "position_scale_mm": POSITION_SCALE_MM,
            "brownian_diffusion_std_mm_per_sqrt_normalized_time": diffusion_std_mm,
            "brownian_diffusion_std_scaled": diffusion_std_mm / POSITION_SCALE_MM,
            "potential_scale_sigma_squared": (
                diffusion_std_mm / POSITION_SCALE_MM
            )
            ** 2,
        },
        "training": {
            "training_seed": args.training_seed,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "spline_first_iterations": args.spline_first_iterations,
            "spline_iterations": args.spline_iterations,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "batch_size": args.batch_size,
            "use_control_variate": args.use_control_variate,
            "num_s_per_t": args.num_s_per_t,
            "hidden_dim": args.hidden_dim,
            "time_embed_dim": args.time_embed_dim,
            "pretrain_refresh_epochs": args.pretrain_refresh_epochs,
            "finetune_refresh_epochs": args.finetune_refresh_epochs,
            "finetune_start_direction": args.finetune_start_direction,
            "pretrain_lr": args.pretrain_lr,
            "finetune_lr": args.finetune_lr,
            "spline_n_mean": args.spline_n_mean,
            "spline_n_std": args.spline_n_std,
            "spline_grid_size": args.spline_grid_size,
            "spline_time_samples": args.spline_time_samples,
            "spline_path_samples": args.spline_path_samples,
            "resume_score_checkpoint": (
                None
                if args.resume_score_checkpoint is None
                else args.resume_score_checkpoint.name
            ),
            "skip_training": args.skip_training,
            "training_seconds": training_seconds,
        },
        "sampling": {
            "sample_seed": args.sample_seed,
            "direction": args.sample_direction,
            "n_samples": args.n_samples,
            "logged_times": logged_times.tolist(),
        },
        "exact_target": {
            "n_hypotheses": int(exact.probabilities.size),
            "reference_metrics": reference_metrics,
            "initial_boundary_marginal": exact_initial.tolist(),
            "final_boundary_marginal": exact_final.tolist(),
            "endpoint_permutation_joint": exact_endpoint_joint.tolist(),
        },
        "sampled_target": {
            **comparison.to_dict(),
            "initial_boundary_marginal": sample_initial.tolist(),
            "final_boundary_marginal": sample_final.tolist(),
            "endpoint_permutation_joint": sample_endpoint_joint.tolist(),
        },
        "canonical_sampled_target": canonical_comparison.to_dict(),
        "rao_blackwellized_target": rao_blackwellized_comparison.to_dict(),
        "canonical_rao_blackwellized_target": (
            canonical_rao_blackwellized_comparison.to_dict()
        ),
        "finite_sample_baseline": {
            "labelled": labelled_baseline.to_dict(),
            "canonical": canonical_baseline.to_dict(),
        },
        "continuous_target": {
            "canonical_sliced_wasserstein_um": sliced_wasserstein_um,
            "exact_finite_sample_baseline_um": sliced_wasserstein_baseline_um,
            "n_projections": args.wasserstein_projections,
            "sampled_observation_residuals": residual_summary(
                sampled_residuals_um
            ),
            "exact_observation_residuals": residual_summary(exact_residuals_um),
        },
        "leading_modes": leading_modes,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        run_dir / "samples.npz",
        sampled_paths=sampled_paths.astype(np.float32),
        sampled_starts=sampled_starts.astype(np.float32),
        sampled_associations=sampled_associations.astype(np.int8),
        canonical_sampled_associations=canonical_samples.astype(np.int8),
        exact_associations=exact.associations.astype(np.int8),
        exact_probabilities=exact.probabilities,
        exact_sampled_paths=exact_paths.astype(np.float32),
        empirical_probabilities=empirical,
        rao_blackwellized_probabilities=rao_blackwellized,
        logged_times=logged_times,
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsbm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--regime", choices=tuple(REGIMES), default="ambiguous")
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=10_000)
    parser.add_argument("--sample-direction", choices=("f", "b"), default="f")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-batch-size", type=int, default=512)
    parser.add_argument("--n-samples", type=int, default=4_096)
    parser.add_argument("--baseline-replicates", type=int, default=500)
    parser.add_argument("--wasserstein-projections", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=200)
    parser.add_argument("--spline-first-iterations", type=int, default=200)
    parser.add_argument("--spline-iterations", type=int, default=50)
    parser.add_argument(
        "--use-control-variate", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num-s-per-t", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--time-embed-dim", type=int, default=64)
    parser.add_argument("--spline-n-mean", type=int, default=9)
    parser.add_argument("--spline-n-std", type=int, default=12)
    parser.add_argument("--spline-grid-size", type=int, default=256)
    parser.add_argument("--spline-time-samples", type=int, default=32)
    parser.add_argument("--spline-path-samples", type=int, default=2)
    parser.add_argument("--pretrain-refresh-epochs", type=int)
    parser.add_argument("--finetune-refresh-epochs", type=int, default=100)
    parser.add_argument(
        "--finetune-start-direction", choices=("f", "b"), default="f"
    )
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--resume-score-checkpoint", type=Path)
    parser.add_argument("--resume-control-variate-checkpoint", type=Path)
    parser.add_argument("--resume-optimizer-checkpoint", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
