import json

from sbulm.wp4_existence_calibration import run_wp4_existence_calibration


def test_wp4_existence_calibration_writes_gate_outputs(tmp_path) -> None:
    result = run_wp4_existence_calibration(
        output_dir=tmp_path,
        n_particles=128,
        n_smc_repeats=1,
        mh_sweeps=1,
        final_mh_sweeps=0,
        n_projections=4,
        baseline_replicates=5,
        workers=1,
    )
    assert result["method"]["uses_exact_enumeration_during_inference"] is False
    assert result["method"]["association_states"]["-2"] == "slot absent"
    assert len(result["gate_runs"]) == 4
    assert set(result["gate_components"]) == {
        "distribution_event_path_checks_passed",
        "finite_sample_support_checks_passed",
        "individual_evidence_checks_passed",
        "pooled_evidence_errors_by_regime",
        "pooled_evidence_checks_passed",
    }
    for name in (
        "wp4_existence_calibration.json",
        "wp4_existence_calibration.png",
    ):
        assert (tmp_path / name).is_file()
    with (tmp_path / "wp4_existence_calibration.json").open() as stream:
        saved = json.load(stream)
    assert saved["experiment"] == result["experiment"]
