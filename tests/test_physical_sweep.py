import json

from sbulm.physical_sweep import run_physical_calibration_sweep


def test_small_physical_sweep_writes_auditable_outputs(tmp_path) -> None:
    result = run_physical_calibration_sweep(
        output_dir=tmp_path,
        separations_um=(0.0, 5.0),
        noise_levels_um=(2.5,),
        n_seeds=1,
        sensitivity_seeds=1,
        workers=1,
    )
    assert result["physical_reference"]["displacement_per_frame_um"] == 6.84
    assert (tmp_path / "physical_matern_calibration_map.csv").is_file()
    assert (tmp_path / "physical_matern_speed_sensitivity.csv").is_file()
    assert (tmp_path / "physical_matern_calibration_map.png").is_file()
    with (tmp_path / "physical_matern_calibration.json").open() as stream:
        saved = json.load(stream)
    assert saved["n_seeds_per_map_cell"] == 1
    assert len(saved["metric_means"]["calibrated_true_mass"]) == 1
