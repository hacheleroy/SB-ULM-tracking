import json

from sbulm.smc_calibration import run_smc_calibration


def test_smc_calibration_writes_gate_and_map_outputs(tmp_path) -> None:
    result = run_smc_calibration(
        output_dir=tmp_path,
        n_particles=256,
        n_smc_repeats=1,
        map_n_particles=256,
        map_n_data_seeds=1,
        mh_sweeps=2,
        n_projections=8,
        separations_um=(0.0,),
        noise_levels_um=(5.0,),
        workers=1,
    )
    assert result["method"]["uses_exact_enumeration_during_inference"] is False
    assert len(result["gate_runs"]) == 3
    assert result["map"]["n_evaluations"] == 1
    for name in (
        "smc_matern_calibration.json",
        "smc_matern_calibration.png",
        "smc_matern_calibration_map.csv",
        "smc_matern_calibration_map.png",
    ):
        assert (tmp_path / name).is_file()
    with (tmp_path / "smc_matern_calibration.json").open() as stream:
        saved = json.load(stream)
    assert saved["experiment"] == result["experiment"]
