import json

from sbulm.wp3_null_calibration import run_wp3_null_calibration


def test_wp3_null_calibration_writes_gate_and_map_outputs(tmp_path) -> None:
    result = run_wp3_null_calibration(
        output_dir=tmp_path,
        n_particles=256,
        n_smc_repeats=1,
        map_n_particles=256,
        map_n_data_seeds=1,
        mh_sweeps=2,
        final_mh_sweeps=1,
        n_projections=8,
        baseline_replicates=10,
        map_baseline_replicates=10,
        detection_probabilities=(0.80,),
        clutter_rates=(0.35,),
        workers=1,
    )
    assert result["method"]["uses_exact_enumeration_during_inference"] is False
    assert len(result["gate_runs"]) == 4
    assert result["map"]["n_evaluations"] == 1
    for name in (
        "wp3_null_calibration.json",
        "wp3_null_calibration.png",
        "wp3_null_calibration_map.csv",
        "wp3_null_calibration_map.png",
    ):
        assert (tmp_path / name).is_file()
    with (tmp_path / "wp3_null_calibration.json").open() as stream:
        saved = json.load(stream)
    assert saved["experiment"] == result["experiment"]
