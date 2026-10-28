# TDD evidence: release detection (Phase 1)

Source plan: in-session plan "release detection, then history" (2026-10-28), Phase 1.

## User journeys
1. A student filming a drop wants the frames where the ball is still in the hand ignored, so g is not biased.
2. They want to see what was dropped (a note, plus grey points on the video).
3. They want to be able to turn it off.

## RED / GREEN
- RED 893d98e: `tests/test_release.py` failed at import (`rest_samples` / `trim_rest` missing).
- RED (wiring): server `KeyError: 'notes'`, the agent question lacked "at rest", `--keep-rest` unknown.
- GREEN 8203759: 137 passed.
- Review round (ecc:mle-reviewer, ecc:python-reviewer, ecc:silent-failure-hunter): new tests
  (Gaussian jitter at 300 px/m, interlaced field jitter, outlier first sample, long hold with a
  3-sample blip, tiny tracks, NaN, counts, pendulum turning points, bounce, CLI JSON counts) were RED
  first, then GREEN after the fixes. Final: `.venv/bin/pytest -q -W ignore::scipy.optimize.OptimizeWarning`
  gives 151 passed.

## What the tests guarantee
| # | Guarantee | Test |
|---|-----------|------|
| 1 | A held ball is trimmed and g comes back within 2 % and passes the checks | test_held_ball_is_trimmed_and_g_recovers |
| 2 | Without trimming the same data biases g low (the bug) | test_held_samples_bias_g_without_trimming |
| 3 | A clean release, and a 240 fps start that is already moving, are not trimmed | test_clean_release_is_not_trimmed, test_fast_camera_moving_start_is_not_trimmed |
| 4 | Pendulum turning points (3, 10, 20 cm amplitude) are never trimmed | test_pendulum_turning_points_are_not_trimmed |
| 5 | A ball lying still after landing is trimmed; a bounce back onto the same spot is not | test_ball_lying_still_after_landing_is_trimmed, test_bounce_back_to_the_rest_spot_is_not_a_landing |
| 6 | It is robust to 1 px Gaussian jitter at 300 px/m, interlaced half-line jitter, and one outlier first sample | test_gaussian_jitter_at_coarse_scale_still_trims, test_interlaced_field_jitter_still_trims, test_outlier_first_sample_does_not_hide_the_hold |
| 7 | It never leaves fewer than 8 samples to fit (loophole closed); tiny tracks and NaN are safe | test_long_hold_with_a_short_blip_is_left_alone, test_tiny_tracks_are_left_alone, test_nan_ends_a_still_run |
| 8 | Times are not shifted (overlay sync) | test_kept_track_keeps_original_times |
| 9 | CLI: notes print, `--keep-rest` disables it, `--json` has `rest_dropped` | test_cli_* |
| 10 | API: notes plus dropped points in the track event; `detect_release=false` disables it; the agent is told | test_ball_held_before_release_is_dropped_and_shown, test_release_detection_can_be_turned_off, test_agent_is_told_about_the_dropped_samples |
| 11 | Deterministic fallback picks the fewest failed checks, then the best R² | test_choose_prefers_fewest_failed_checks, test_choose_falls_back_to_best_r2_not_smallest_rmse |

## Regression on real clips (deterministic CLI)
| Clip | --keep-rest | trimmed |
|------|-------------|---------|
| BallDrop, recipe search area | g 9.728 ok | unchanged |
| BallDrop, user search area (top covers the hand) | g 9.016 flagged | 11 dropped, g 9.624 ok |
| BallDrop, no search area | g 7.932 flagged | 7 dropped, g 9.698 ok |
| BallTossOut | projectile g 10.015 flagged (perspective) | unchanged |
| Commons pendulum | L 0.590 m | unchanged |
| Tracker Pendulum, default settings | flagged, bad tracking | still flagged |

## Known gaps
- A ball moving with the hand during a throw has no still phase, so it is not trimmed. The agent and the checks handle it.
- A hand drifting more than about 1 cm while holding ends the still run early, so only part of the hold is trimmed.
- The agent only learns about the trim through the question sentence; there is no tool field for it.
