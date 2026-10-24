import json

from sbulm.wp2_likelihood import run_wp2_likelihood_sweep


def test_small_wp2_sweep_writes_auditable_outputs(tmp_path) -> None:
    result = run_wp2_likelihood_sweep(
        output_dir=tmp_path,
        separations_um=(0.0, 5.0),
        noise_levels_um=(5.0,),
        n_seeds=1,
        workers=1,
    )
    assert result["n_evaluations"] == 2
    assert result["prior"]["length_scale_ms"] > 0
    assert (tmp_path / "wp2_likelihood_map.csv").is_file()
    assert (tmp_path / "wp2_credible_coverage.csv").is_file()
    assert (tmp_path / "wp2_likelihood_map.png").is_file()
    assert (tmp_path / "wp2_credible_coverage.png").is_file()
    with (tmp_path / "wp2_localization_likelihood.json").open() as stream:
        saved = json.load(stream)
    assert saved["n_seeds_per_cell"] == 1
    assert len(saved["global_credible_coverage"]) == 5
