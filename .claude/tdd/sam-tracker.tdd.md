# TDD evidence: SAM 3.1 tracker

Source plan: the inline `/plan` for SAM 3.1 approved on 2026-10-28 (Task 0 real probe, Tasks 1-5).
Branch: `feat/sam`.

## Task 0: real probe (before any code)

| Probe | Result | Cost |
|---|---|---|
| Upload BallDrop frames re-encoded as MP4 (`POST /v1/files`) | 200 in 2.4 s, `file-…`, `expires_at: null` | — |
| Text prompt `"ball"` | 200 in 2.6 s, stream completed, **no frame lines** | $0.01 (50-frame minimum) |
| Text prompt `"small white ball"` | no frame lines | $0.01 |
| Box prompt `<0f>1<|box;…|>`, forward | 18 frames tracked, confidence 0.73–0.79 | $0.01 |

Findings that changed the plan: text prompts miss a small ball, so the box drawn on the first frame
is the main mode, and text is a fallback. `usage.video_frames_processed` reports the billed count
(50 for a 33-frame clip), so it cannot verify the frame mapping. The streamed responses were kept as
`tests/fixtures/sam/*.sse`.

## User journeys

- As a student, I box the ball on the first frame and SAM follows it, so the hand that drops it is never tracked.
- As a student without a key, or when the API fails or the budget is used up, I still get a measurement from
  the motion tracker, and a note tells me why.
- As the owner, every SAM call is priced and logged before it can push spend over `PHYSICSLENS_MAX_USD`.

## RED / GREEN checkpoints

| Stage | Commit | Evidence |
|---|---|---|
| RED: tracker and client | 98107d1 | collection error, `No module named 'physicslens.llm.sam_client'` |
| GREEN | f04dd58 | `tests/test_sam.py` 25 passed; full suite 177 passed |
| RED: wiring | 668c3c0 | 7 failed (TrackSettings has no `sam_box`, no `make_sam_client`) |
| GREEN | 4584248 | full suite 185 passed |
| UI | f7452d6 | `tsc` and `npm run build` pass; Chrome E2E below |
| RED: review findings | test commit after f7452d6 | collection error, `cannot import name 'SamTrackError'` |
| GREEN | fix commit after it | full suite 202 passed |

## Test specification

| # | What is guaranteed | Test |
|---|---|---|
| 1 | A recorded real stream parses to 18 frames with the right box | `test_recorded_stream_gives_one_line_per_tracked_frame` |
| 2 | Out-of-order frames, gaps, sparse ids and garbage lines are handled | `test_frames_out_of_order_with_gaps_and_sparse_ids` |
| 3 | A failed or cut stream is an error, and errors are redacted | `test_failed_stream_is_an_error`, `test_stream_without_completion_is_an_error`, `test_stream_errors_are_redacted` |
| 4 | Box prompt grammar is correct and clipped to the frame | `test_box_prompt_grammar`, `test_box_prompt_is_clipped_to_the_frame` |
| 5 | Track times are `video.timestamps[N]`; positions are box centres | `test_track_sam_with_a_box_uses_box_centres_and_video_times` |
| 6 | A frame index past the clip is an error, not dropped silently | `test_frame_past_the_clip_means_the_mapping_is_off` |
| 7 | Windows over 900 frames are refused before paying | `test_too_long_clip_is_refused_before_paying` |
| 8 | The boxed object is followed; a moving object beats a still one seen more often | `test_pick_with_a_box_follows_the_boxed_object`, `test_pick_prefers_motion_over_a_static_object_seen_more_often` |
| 9 | Upload then stream, with the key only in the header, to `api.meta.ai` | `test_segment_uploads_then_streams` |
| 10 | Cost: 50-frame minimum, an estimate when usage is missing or NaN, and an estimate when the request fails after upload | `test_segment_logs_frames_at_the_minimum_charge`, `test_missing_usage_is_estimated_from_the_frame_count`, `test_non_finite_usage_is_estimated_not_a_crash`, `test_failure_after_upload_still_logs_an_estimate` |
| 11 | Budget includes the call about to be made | `test_budget_stops_before_upload`, `test_budget_counts_the_call_about_to_be_made` |
| 12 | Cache hits cost nothing; misses and corrupt entries are not reused | `test_segment_is_cached`, `test_empty_result_is_not_cached`, `test_corrupt_cache_entry_is_dropped_and_refetched` |
| 13 | 429 is retried; 400 is permanent; network failures give up | `test_rate_limit_is_retried`, `test_bad_request_is_permanent_and_key_is_redacted`, `test_network_failure_gives_up` |
| 14 | When SAM fails, motion takes over with a note naming the cause (budget included) | `test_sam_failure_falls_back_to_motion_and_says_so`, `test_budget_fallback_says_it_is_the_budget`, `test_api_sam_fallback_is_visible` |
| 15 | The API tracks with SAM and reports the tracker used | `test_api_tracks_with_sam_and_reports_the_tracker` |
| 16 | The API rejects raw prompt tokens, empty prompts and bad boxes | `test_api_rejects_bad_sam_settings`, `test_api_rejects_an_overlong_prompt` |
| 17 | The CLI tracks with SAM and refuses `--sam-*` without `--tracker sam` | `test_cli_tracks_with_sam`, `test_cli_sam_options_need_the_sam_tracker` |

## Real runs (Meta Model API)

| Clip | Before | SAM 3.1 |
|---|---|---|
| BallDrop, CLI, box on frame 0 | 9.728 (tuned colour recipe), 9.624 (motion with search area) | 9.653 ± 0.043, checks pass, no search area |
| BallDrop, Chrome UI, box drawn by hand | — | 9.735 ± 0.039, all checks pass |
| BallTossOut, `--start 0.33`, box | 10.015, flagged (perspective) | 10.038, flagged (same perspective) |
| Commons pendulum, text `"pendulum bob"` | — | nothing found; fell back to motion, with the note |
| Commons pendulum, box | L = 0.590 (hand-tuned HSV) | L = 0.590, 105/105 frames, no tuning |
| BallDrop rerun after the fixes | — | same g, cache hit, no new cost row |

Total real SAM spend for this feature: about $0.08.

## Coverage and known gaps

`pytest --cov`: total 92%; `track/sam.py` 97%, `llm/sam_client.py` 90%.

- **Not verified:** SAM decodes exactly our frames. Usage reports the billed count, so the only check
  is that no index falls past the clip. On real clips, the frame counts matched the tracked indices.
- **Possible bias:** the box centre includes interlace smear and partial occlusion. On BallDrop this
  shifts the fitted t0, not g. Mask centroids (base85 masks) are not decoded.
- **Edge boxes are not dropped:** the Commons pendulum bob is always cut by the frame edge and still
  measures correctly.
- **Deferred to the public deploy phase:**
  - per-IP quota;
  - budget persisted across restarts;
  - mandatory budget in public mode;
  - deleting uploaded files from the Files API;
  - response size cap;
  - generic client-facing error text.
