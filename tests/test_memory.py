"""Long and high-resolution clips fit in memory: masks stay int16 and are made once, long windows are
refused before they are decoded, and the CLI shrinks frames like the UI does."""

import tracemalloc

import cv2
import numpy as np
import pytest

import cli
from physicslens import pipeline
from physicslens.ingest import load_video
from physicslens.physics import models
from physicslens.pipeline import TrackSettings, track
from physicslens.track import autodetect, motion
from physicslens.track.motion import FLICKER_SHARE, moving_masks
from tests.test_autodetect import constant_speed_video
from tests.test_robustness import PX_PER_M, write_drop


def reference_masks(video, threshold=40):
    """The float64 formula the int16 one must match."""
    gray = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in video.frames], np.int16)
    moving = (np.abs(gray - np.median(gray, axis=0)) > threshold).astype(np.uint8)
    return moving & (moving.mean(axis=0) <= FLICKER_SHARE).astype(np.uint8)


@pytest.mark.parametrize("n_frames", [27, 28])  # an even count gives half-number medians
def test_masks_are_unchanged(tmp_path, n_frames):
    write_drop(tmp_path / "d.avi", 60, 60, n_frames, distractor=True)
    video = load_video(str(tmp_path / "d.avi"))
    noisy = [np.clip(f.astype(np.int16) + np.random.default_rng(i).integers(-45, 46, f.shape), 0, 255).astype(np.uint8)
             for i, f in enumerate(video.frames)]  # noise near the threshold exercises the rounding
    video.frames[:] = noisy
    for threshold in (40, 41):
        assert np.array_equal(moving_masks(video, threshold), reference_masks(video, threshold))


def test_masks_never_hold_a_float_copy_of_the_clip(tmp_path):
    write_drop(tmp_path / "d.avi", 60, 60, 80)
    video = load_video(str(tmp_path / "d.avi"))
    stack = len(video.frames) * video.height * video.width * 2  # the int16 gray stack
    tracemalloc.start()
    moving_masks(video)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # int16 stack + median's partition copy + uint8 masks; one float64 copy alone is 4 stacks.
    assert peak < 3.5 * stack


def test_a_too_long_window_is_refused_before_it_is_decoded(tmp_path):
    write_drop(tmp_path / "d.avi", 60, 60, 50)
    with pytest.raises(ValueError, match="set a start and end"):
        load_video(str(tmp_path / "d.avi"), max_frames=30)
    assert len(load_video(str(tmp_path / "d.avi"), max_frames=50).frames) == 50


def test_the_masks_are_made_once_per_motion_run(tmp_path, monkeypatch):
    # A failed search, then the field-order passes: one median background for all of them.
    calls = []
    real = motion.moving_masks

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    for module in (motion, autodetect, pipeline):
        monkeypatch.setattr(module, "moving_masks", counted, raising=False)
    video = load_video(constant_speed_video(tmp_path / "c.avi"))
    track(video, TrackSettings("motion", fields="auto", px_per_m=PX_PER_M))
    assert len(calls) == 1


def test_shared_masks_are_not_changed_by_a_search_area(tmp_path):
    write_drop(tmp_path / "d.avi", 60, 60, 27)
    video = load_video(str(tmp_path / "d.avi"))
    masks = moving_masks(video)
    before = masks.copy()
    autodetect.find_ball(video, PX_PER_M, roi=(0, 0, 400, 479), moving=masks)
    with pytest.raises(ValueError):
        motion.track_motion(video, roi=(0, 0, 100, 100), moving=masks)  # nothing moves there
    assert np.array_equal(masks, before)


def wide_drop(path):
    """A 1920 px wide drop: the CLI shrinks it to 960 px, settings and scale with it."""
    write_drop(path, 60, 60, 27, size=(1920, 1080))
    return str(path)


def test_the_cli_shrinks_wide_frames_and_scales_its_settings(tmp_path, capsys):
    import json

    path = wide_drop(tmp_path / "w.avi")
    g = {}
    for width in ("960", "0"):  # 0 = full resolution
        assert cli.main([path, "--px-per-m", str(PX_PER_M), "--max-width", width, "--roi", "200,0,450,1079", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        best = next(r for r in report["fits"] if r["fit_id"] == report["best"])
        g[width] = best["params"]["g"]
    assert g["960"] == pytest.approx(models.G, rel=0.02) and g["960"] == pytest.approx(g["0"], rel=0.01)
