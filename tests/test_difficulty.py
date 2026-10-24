import json

from sbulm.difficulty import run_difficulty_sweep


def test_small_difficulty_sweep_writes_expected_outputs(tmp_path) -> None:
    result = run_difficulty_sweep(
        output_dir=tmp_path,
        separations=(0.0, 0.1),
        noise_levels=(0.025,),
        n_seeds=1,
    )
    assert result["experiment"] == "crossing_difficulty_map"
    assert len(result["metric_means"]["true_hypothesis_mass"]) == 1
    assert len(result["metric_means"]["true_hypothesis_mass"][0]) == 2
    assert (tmp_path / "difficulty_map.csv").is_file()
    assert (tmp_path / "difficulty_map.png").is_file()
    with (tmp_path / "difficulty_map.json").open(encoding="utf-8") as stream:
        saved = json.load(stream)
    assert saved["n_seeds"] == 1

