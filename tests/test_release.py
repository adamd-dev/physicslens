"""Release detection: samples where the ball sits still (held before release, lying after landing) are not
part of the free motion and are dropped before fitting, with a note saying so."""

import numpy as np
import pytest

from physicslens.agent.tools import Toolbox
from physicslens.calibrate import Calibration
from physicslens.physics import models
from physicslens.pipeline import rest_samples, run_fits, to_track_data, trim_rest
from physicslens.track.base import Track

PX_PER_M = 400


def held_drop(fps=60, hold_s=0.2, fall_s=0.45, rest_after_s=0.0, v0=0.0, seed=0):
    """Ball held still (±0.7 px hand jitter), released from rest at hold_s, optionally lying still after."""
    rng = np.random.default_rng(seed)
    dt = 1 / fps
    t_hold = np.arange(0, hold_s - 1e-9, dt)
    t_fall = np.arange(0, fall_s, dt)
    y_fall = 0.1 + models.free_fall(t_fall, 0, v0, models.G)
    t_rest = np.arange(dt, rest_after_s + 1e-9, dt)
    t = np.concatenate([t_hold, hold_s + t_fall, hold_s + t_fall[-1] + t_rest])
    y = np.concatenate([np.full(len(t_hold), 0.1), y_fall, np.full(len(t_rest), y_fall[-1])])
    x = np.full(len(t), 0.5)
    still = np.r_[np.ones(len(t_hold), bool), np.zeros(len(t_fall), bool), np.ones(len(t_rest), bool)]
    jitter = rng.uniform(-0.7, 0.7, (2, len(t))) * still
    return Track(t, x * PX_PER_M + jitter[0], y * PX_PER_M + jitter[1], "test"), len(t_hold), len(t_rest)


def best_free_fall(tr):
    box = Toolbox(to_track_data(tr, Calibration(PX_PER_M)))
    return next(r for r in run_fits(box, "free_fall") if r["name"] == "free_fall")


def test_held_ball_is_trimmed_and_g_recovers():
    tr, n_hold, _ = held_drop()
    lead, trail = rest_samples(tr, PX_PER_M)
    # The first one or two samples after release have barely moved (0.5 g t² < 5 mm): they go too.
    assert n_hold <= lead <= n_hold + 3
    assert trail == 0
    trim = trim_rest(tr, PX_PER_M)
    kept, notes = trim.track, trim.notes
    fit = best_free_fall(kept)
    assert fit["params"]["g"] == pytest.approx(models.G, rel=0.02)
    assert fit["check"]["ok"]
    assert len(notes) == 1 and f"{lead} samples" in notes[0]


def test_held_samples_bias_g_without_trimming():
    # The bug this fixes: the fit bends through the held points and reports a low g.
    tr, _, _ = held_drop()
    assert best_free_fall(tr)["params"]["g"] < models.G * 0.97


def test_clean_release_is_not_trimmed():
    tr, _, _ = held_drop(hold_s=0)
    assert rest_samples(tr, PX_PER_M) == (0, 0)
    trim = trim_rest(tr, PX_PER_M)
    assert trim.track is tr and trim.notes == []


def test_fast_camera_moving_start_is_not_trimmed():
    # 240 fps: three samples fit inside 5 mm, but 8 ms of motion is not a held ball.
    tr, _, _ = held_drop(fps=240, hold_s=0, v0=1.0)
    assert rest_samples(tr, PX_PER_M) == (0, 0)


@pytest.mark.parametrize("amplitude", [0.03, 0.1, 0.2])
def test_pendulum_turning_points_are_not_trimmed(amplitude):
    # The bob is slow near a turning point, but it comes back there every swing: not a held ball.
    t = np.arange(0, 4, 1 / 30)
    omega = np.sqrt(models.G / 0.6)
    tr = Track(t, (0.5 + amplitude * np.cos(omega * t)) * PX_PER_M, np.full(len(t), 0.9 * PX_PER_M), "test")
    assert rest_samples(tr, PX_PER_M) == (0, 0)


def test_bounce_back_to_the_rest_spot_is_not_a_landing():
    # Ball sits on the floor, gets kicked up and lands back on the same spot: keep it all.
    t = np.arange(0, 1.2, 1 / 60)
    up = np.clip(2.0 * (t - 0.3) - 0.5 * models.G * (t - 0.3) ** 2, 0, None) * (t > 0.3)
    tr = Track(t, np.full(len(t), 200.0), (1.0 - up) * PX_PER_M, "test")
    assert rest_samples(tr, PX_PER_M)[0] == 0


def test_ball_lying_still_after_landing_is_trimmed():
    tr, _, n_rest = held_drop(hold_s=0, rest_after_s=0.3)
    lead, trail = rest_samples(tr, PX_PER_M)
    assert lead == 0
    assert n_rest <= trail <= n_rest + 1
    trim = trim_rest(tr, PX_PER_M)
    kept = trim.track
    assert len(kept) == len(tr) - trail
    assert "after" in trim.notes[0]
    assert best_free_fall(kept)["check"]["ok"]


def test_nothing_moving_is_left_alone():
    t = np.arange(0, 0.5, 1 / 60)
    tr = Track(t, np.full(len(t), 200.0), np.full(len(t), 100.0), "test")
    assert rest_samples(tr, PX_PER_M) == (0, 0)
    assert trim_rest(tr, PX_PER_M).track is tr


def test_gaussian_jitter_at_coarse_scale_still_trims():
    # 300 px/m with 1 px Gaussian jitter: a 1.5 px radius would stop the run after a sample or two.
    rng = np.random.default_rng(1)
    t = np.arange(0, 0.65, 1 / 60)
    held = t < 0.2
    y = np.where(held, 0.1, 0.1 + 0.5 * models.G * (t - 0.2) ** 2) * 300 + rng.normal(0, 1, len(t)) * held
    x = 150 + rng.normal(0, 1, len(t)) * held
    lead, _ = rest_samples(Track(t, x, y, "test"), 300)
    assert lead >= held.sum() - 1


def test_interlaced_field_jitter_still_trims():
    # Alternating fields put a still ball half a line up and down.
    t = np.arange(0, 0.6, 1 / 60)
    held = t < 0.2
    y = np.where(held, 40 + 0.5 * (-1) ** np.arange(len(t)), 40 + 0.5 * models.G * (t - 0.2) ** 2 * PX_PER_M)
    lead, _ = rest_samples(Track(t, np.full(len(t), 200.0), y, "test"), PX_PER_M)
    assert lead >= held.sum()


def test_outlier_first_sample_does_not_hide_the_hold():
    tr, n_hold, _ = held_drop()
    tr.x[0] += 6  # one bad centroid
    assert rest_samples(tr, PX_PER_M)[0] >= n_hold - 1


def test_long_hold_with_a_short_blip_is_left_alone():
    # 1 s still then a 3-sample glitch: trimming would leave too few points to fit honestly.
    t = np.arange(0, 1.05, 1 / 60)
    y = np.full(len(t), 100.0)
    y[-3:] += [20, 45, 80]
    assert rest_samples(Track(t, np.full(len(t), 200.0), y, "test"), PX_PER_M) == (0, 0)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_tiny_tracks_are_left_alone(n):
    t = np.arange(n) / 60
    assert rest_samples(Track(t, np.zeros(n), np.zeros(n), "test"), PX_PER_M) == (0, 0)


def test_nan_ends_a_still_run():
    tr, _, _ = held_drop()
    tr.y[3] = np.nan
    assert rest_samples(tr, PX_PER_M)[0] <= 3


def test_trim_reports_counts():
    tr, n_hold, _ = held_drop(rest_after_s=0.3)
    trim = trim_rest(tr, PX_PER_M)
    assert trim.lead >= n_hold and trim.trail > 0
    assert len(trim.track) == len(tr) - trim.lead - trim.trail


def test_kept_track_keeps_original_times():
    # The overlay maps samples back to the video clock: trimming must not shift time.
    tr, _, _ = held_drop()
    kept = trim_rest(tr, PX_PER_M).track
    lead = len(tr) - len(kept)
    assert np.array_equal(kept.t, tr.t[lead:])
    assert kept.source == tr.source


def held_drop_video(tmp_path):
    from tests.test_robustness import write_drop

    path = tmp_path / "held.avi"
    write_drop(path, capture_fps=60, playback_fps=60, n_frames=39, hold_frames=12)
    return str(path)


def test_cli_trims_and_says_so(tmp_path, capsys):
    import cli
    from tests.test_robustness import PX_PER_M as VIDEO_PX_PER_M

    assert cli.main([held_drop_video(tmp_path), "--px-per-m", str(VIDEO_PX_PER_M), "--tracker", "color"]) == 0
    out = capsys.readouterr().out
    assert "warning: dropped" in out and "before the motion" in out


def test_cli_keep_rest_turns_it_off(tmp_path, capsys):
    import cli
    from tests.test_robustness import PX_PER_M as VIDEO_PX_PER_M

    assert cli.main([held_drop_video(tmp_path), "--px-per-m", str(VIDEO_PX_PER_M), "--tracker", "color", "--keep-rest"]) == 1
    assert "dropped" not in capsys.readouterr().out


def test_cli_json_has_structured_counts(tmp_path, capsys):
    import json

    import cli
    from tests.test_robustness import PX_PER_M as VIDEO_PX_PER_M

    cli.main([held_drop_video(tmp_path), "--px-per-m", str(VIDEO_PX_PER_M), "--tracker", "color", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["rest_dropped"]["start"] >= 12 and report["rest_dropped"]["end"] == 0


def test_agent_question_is_the_same_for_cli_and_api():
    from physicslens.pipeline import agent_question

    assert agent_question() == "Measure the physics in this tracked trajectory."
    with_height = agent_question(0.9, ["dropped 3 samples"])
    assert "0.9 m above" in with_height and "(dropped 3 samples) were already removed" in with_height
