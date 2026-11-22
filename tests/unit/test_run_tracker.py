import pytest

import run_tracker


def test_cli_runs_tracker_with_video_and_custom_options(tmp_path, monkeypatch):
    video = tmp_path / "sample.mp4"
    video.touch()
    output_dir = tmp_path / "results"
    captured = {}

    class FakeTracker:
        def __init__(self, video_path, **kwargs):
            captured["video_path"] = video_path
            captured.update(kwargs)

        def run(self, display):
            captured["display"] = display

    monkeypatch.setattr(run_tracker, "Tracker", FakeTracker)

    run_tracker.main([
        str(video),
        "--output-dir", str(output_dir),
        "--class-id", "2",
        "--confidence", "0.7",
        "--horizons", "0.5", "1.5",
        "--show",
    ])

    assert captured["video_path"] == str(video)
    assert captured["model_path"] == "yolo11n.pt"
    assert captured["object_class"] == 2
    assert captured["confidence"] == 0.7
    assert captured["horizons"] == [0.5, 1.5]
    assert captured["display"] is True
    assert captured["output_video"] == str(output_dir / "output.mp4")
    assert output_dir.is_dir()


def test_cli_rejects_missing_video(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        run_tracker.main([str(tmp_path / "missing.mp4")])


def test_cli_rejects_non_positive_horizon(tmp_path):
    video = tmp_path / "sample.mp4"
    video.touch()

    with pytest.raises(SystemExit, match="2"):
        run_tracker.main([str(video), "--horizons", "0"])


def test_cli_rejects_non_finite_horizon(tmp_path):
    video = tmp_path / "sample.mp4"
    video.touch()

    with pytest.raises(SystemExit, match="2"):
        run_tracker.main([str(video), "--horizons", "nan"])
