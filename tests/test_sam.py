"""SAM 3.1 tracker offline: recorded streams and a fake transport stand in for the Meta Model API."""

import json
from pathlib import Path

import numpy as np
import pytest

from physicslens.ingest import Video, load_video
from physicslens.llm.meta_client import (
    BudgetExceededError,
    LLMRequestError,
    LLMResponseError,
    LLMTransportError,
)
from physicslens.llm.sam_client import SamClient, stream_text
from physicslens.track.sam import (
    SamTrackError,
    box_prompt,
    encode_clip,
    parse_frames,
    pick_object,
    track_sam,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sam"
KEY = "sk-test-SECRET-123"


def box(i, x1, y1, x2, y2, c=None):
    conf = f";c={c}" if c is not None else ""
    return f"{i}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w=640;h=480{conf}|><|mask;x=0;y=0;data=9,9,<~abc~>|>"


# --- parsing -----------------------------------------------------------------------


def test_recorded_stream_gives_one_line_per_tracked_frame():
    text = stream_text((FIXTURES / "balldrop_box.sse").read_bytes())
    frames = parse_frames(text)
    assert sorted(frames) == list(range(18))
    assert frames[0][1] == (194, 41, 207, 53)


def test_no_match_stream_is_empty_text():
    assert stream_text((FIXTURES / "no_match.sse").read_bytes()) == ""


def test_failed_stream_is_an_error():
    raw = b'event: response.failed\ndata: {"type":"response.failed","response":{"error":{"message":"bad video"}}}\n\n'
    with pytest.raises(LLMResponseError, match="bad video"):
        stream_text(raw)


def test_stream_without_completion_is_an_error():
    raw = b'data: {"type":"response.output_text.delta","delta":"<0f>"}\n\n'
    with pytest.raises(LLMResponseError, match="ended before"):
        stream_text(raw)


def test_frames_out_of_order_with_gaps_and_sparse_ids():
    text = "\n".join([f"<5f>{box(7, 10, 10, 20, 20)}", f"<1f>{box(7, 0, 0, 10, 10, 0.9)},{box(3, 100, 100, 140, 140)}", "garbage", ""])
    frames = parse_frames(text)
    assert sorted(frames) == [1, 5]
    assert frames[1] == {7: (0, 0, 10, 10), 3: (100, 100, 140, 140)}


# --- choosing the object -------------------------------------------------------------


def test_pick_prefers_the_object_seen_most_then_the_one_that_moves():
    frames = {n: {1: (0, 0, 10, 10), 2: (0, 10 * n, 10, 10 * n + 10)} for n in range(10)}
    frames[0][3] = (300, 300, 310, 310)  # one-frame blip
    assert pick_object(frames) == 2


def test_pick_respects_the_search_area():
    frames = {n: {1: (0, 10 * n, 10, 10 * n + 10), 2: (400, 10 * n, 410, 10 * n + 10)} for n in range(10)}
    assert pick_object(frames, roi=(300, 0, 640, 480)) == 2


def test_pick_prefers_motion_over_a_static_object_seen_more_often():
    frames = {n: {1: (0, 0, 10, 10)} for n in range(20)}  # a ball lying in the background, always seen
    for n in range(12):
        frames[n][2] = (100, 10 * n, 110, 10 * n + 10)  # the falling ball, lost after 12 frames
    assert pick_object(frames) == 2


def test_pick_with_a_box_follows_the_boxed_object():
    frames = {n: {5: (200, 10 * n, 210, 10 * n + 10)} for n in range(8)}
    for n in range(20):
        frames.setdefault(n, {})[9] = (0, 10 * n, 10, 10 * n + 10)  # another mover, seen longer
    assert pick_object(frames, prompt_box=(198, 0, 212, 12)) == 5


def test_pick_with_a_box_and_nothing_under_it_is_none():
    frames = {n: {3: (400, 10 * n, 410, 10 * n + 10)} for n in range(10)}
    assert pick_object(frames, prompt_box=(0, 0, 20, 20)) is None


def test_pick_with_nothing_inside_the_area_is_none():
    assert pick_object({0: {1: (0, 0, 10, 10)}}, roi=(300, 300, 400, 400)) is None


# --- tracking ------------------------------------------------------------------------


def synthetic_video(n=12, fps=30.0) -> Video:
    frames = [np.zeros((480, 640, 3), np.uint8) for _ in range(n)]
    return Video(frames, np.arange(n) / fps, fps, 640, 480, "index", 1.0)


class FakeSam:
    def __init__(self, text):
        self.text, self.calls = text, []

    def segment(self, clip: bytes, prompt: str, direction: str = "both", frames: int | None = None) -> str:
        self.calls.append((clip, prompt, direction))
        self.frames = frames
        return self.text


def test_box_prompt_grammar():
    assert box_prompt((10.4, 20, 30, 40.6), 640, 480) == "<0f>1<|box;x1=10;y1=20;x2=30;y2=41;w=640;h=480|>"


def test_box_prompt_is_clipped_to_the_frame():
    assert box_prompt((-5, -5, 700, 500), 640, 480) == "<0f>1<|box;x1=0;y1=0;x2=639;y2=479;w=640;h=480|>"


def test_track_sam_with_a_box_uses_box_centres_and_video_times():
    video = synthetic_video()
    text = "\n".join(f"<{n}f>{box(1, 100, 10 * n, 110, 10 * n + 10)}" for n in reversed(range(12)) if n != 4)
    sam = FakeSam(text)
    tr = track_sam(video, sam, box=(95, 0, 115, 12))
    _, prompt, direction = sam.calls[0]
    assert prompt.startswith("<0f>1<|box;x1=95;y1=0;") and direction == "forward"
    assert tr.source == "sam" and len(tr) == 11
    assert np.allclose(tr.t, np.delete(video.timestamps, 4))
    assert np.allclose(tr.x, 105) and tr.y[0] == 5 and tr.y[-1] == 115


def test_track_sam_with_text_tracks_both_ways():
    sam = FakeSam("\n".join(f"<{n}f>{box(2, 0, n, 10, n + 10)}" for n in range(12)))
    tr = track_sam(synthetic_video(), sam, prompt="ball")
    assert sam.calls[0][1:] == ("ball", "both") and len(tr) == 12


def test_frame_past_the_clip_means_the_mapping_is_off():
    # SAM frame N must be our frame N; an index past the clip means it decoded a different frame count.
    sam = FakeSam("\n".join(f"<{n}f>{box(1, 0, n, 10, n + 10)}" for n in range(15)))
    with pytest.raises(SamTrackError, match="beyond"):
        track_sam(synthetic_video(), sam, prompt="ball")


def test_frame_count_is_passed_for_the_cost_estimate():
    sam = FakeSam("\n".join(f"<{n}f>{box(1, 0, n, 10, n + 10)}" for n in range(12)))
    track_sam(synthetic_video(), sam, prompt="ball")
    assert sam.frames == 12


def test_too_long_clip_is_refused_before_paying(monkeypatch):
    import physicslens.track.sam as sam_module

    monkeypatch.setattr(sam_module, "MAX_SAM_FRAMES", 10)
    sam = FakeSam("")
    with pytest.raises(SamTrackError, match="12 frames"):
        track_sam(synthetic_video(), sam, prompt="ball")
    assert sam.calls == []


def test_track_sam_with_no_match_says_what_to_do():
    with pytest.raises(ValueError, match="draw a box"):
        track_sam(synthetic_video(), FakeSam(""), prompt="ball")


def test_track_sam_with_too_few_points_fails():
    sam = FakeSam("\n".join(f"<{n}f>{box(1, 0, n, 10, n + 10)}" for n in range(3)))
    with pytest.raises(ValueError, match="3 frames"):
        track_sam(synthetic_video(), sam, box=(0, 0, 10, 10))


def test_track_sam_needs_a_box_or_a_prompt():
    with pytest.raises(ValueError, match="box or a text prompt"):
        track_sam(synthetic_video(), FakeSam(""))


def test_encoded_clip_keeps_every_frame(tmp_path):
    clip = encode_clip(synthetic_video(n=9))
    path = tmp_path / "clip.mp4"
    path.write_bytes(clip)
    back = load_video(str(path))
    assert len(back.frames) == 9 and (back.width, back.height) == (640, 480)


# --- client ----------------------------------------------------------------------------


UPLOAD_OK = (200, {}, json.dumps({"id": "file-42", "status": "uploaded"}).encode())


def stream_ok(frames_processed=50):
    events = [
        {"type": "response.output_text.delta", "delta": "<0f>" + box(1, 0, 0, 10, 10) + "\n"},
        {"type": "response.output_text.delta", "delta": "<1f>" + box(1, 0, 5, 10, 15)},
        {"type": "response.completed", "response": {"usage": {"video_frames_processed": frames_processed}}},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return 200, {"Content-Type": "text/event-stream"}, body.encode()


class FakePost:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": body})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(tmp_path, *responses, **kw):
    post = FakePost(*responses)
    defaults = {"cache_dir": tmp_path / "cache", "cost_log": tmp_path / "costs.csv", "post": post, "sleep": lambda s: None}
    return SamClient(KEY, **{**defaults, **kw}), post


def test_segment_uploads_then_streams(tmp_path):
    sam, post = client(tmp_path, UPLOAD_OK, stream_ok())
    text = sam.segment(b"MP4DATA", "ball", "forward")
    assert parse_frames(text)[1][1] == (0, 5, 10, 15)
    upload, responses = post.calls
    assert upload["url"] == "https://api.meta.ai/v1/files" and b"MP4DATA" in upload["body"] and b"user_data" in upload["body"]
    assert upload["headers"]["Authorization"] == f"Bearer {KEY}"
    body = json.loads(responses["body"])
    assert responses["url"] == "https://api.meta.ai/v1/responses" and body["model"] == "sam-3.1" and body["stream"]
    assert body["metadata"]["propagation_direction"] == "forward"
    content = body["input"][0]["content"]
    assert content == [{"type": "input_text", "text": "ball"}, {"type": "input_video", "file_id": "file-42"}]


def test_segment_logs_frames_at_the_minimum_charge(tmp_path):
    sam, _ = client(tmp_path, UPLOAD_OK, stream_ok(frames_processed=12))
    sam.segment(b"x", "ball")
    row = (tmp_path / "costs.csv").read_text().splitlines()[1].split(",")
    assert row[1] == "sam-3.1" and float(row[4]) == pytest.approx(0.01)  # 50-frame minimum at $0.20 / 1000


def test_segment_is_cached(tmp_path):
    sam, post = client(tmp_path, UPLOAD_OK, stream_ok())
    first = sam.segment(b"x", "ball")
    assert sam.segment(b"x", "ball") == first and len(post.calls) == 2
    assert len((tmp_path / "costs.csv").read_text().splitlines()) == 2


def test_empty_result_is_not_cached(tmp_path):
    # A miss may be transient; replaying it for free forever would pin the clip to the fallback.
    empty = (200, {}, (FIXTURES / "no_match.sse").read_bytes())
    sam, post = client(tmp_path, UPLOAD_OK, empty, UPLOAD_OK, empty)
    assert sam.segment(b"x", "ball") == "" and sam.segment(b"x", "ball") == ""
    assert len(post.calls) == 4


def test_corrupt_cache_entry_is_dropped_and_refetched(tmp_path):
    sam, post = client(tmp_path, UPLOAD_OK, stream_ok(), UPLOAD_OK, stream_ok())
    first = sam.segment(b"x", "ball")
    for entry in (tmp_path / "cache").glob("*.txt"):
        entry.write_bytes(b"\xff\xfe not text")
    with pytest.warns(UserWarning, match="cache"):
        assert sam.segment(b"x", "ball") == first
    assert len(post.calls) == 4


def test_missing_usage_is_estimated_from_the_frame_count(tmp_path):
    no_usage = (200, {}, b'data: {"type":"response.completed","response":{}}\n\n')
    sam, _ = client(tmp_path, UPLOAD_OK, no_usage)
    sam.segment(b"x", "ball", frames=120)
    row = (tmp_path / "costs.csv").read_text().splitlines()[1].split(",")
    assert float(row[4]) == pytest.approx(0.024) and row[5] == "estimated"


def test_non_finite_usage_is_estimated_not_a_crash(tmp_path):
    bad = (200, {}, b'data: {"type":"response.completed","response":{"usage":{"video_frames_processed":NaN}}}\n\n')
    sam, _ = client(tmp_path, UPLOAD_OK, bad)
    sam.segment(b"x", "ball", frames=60)
    assert (tmp_path / "costs.csv").read_text().splitlines()[1].endswith(",estimated")


def test_budget_counts_the_call_about_to_be_made(tmp_path):
    (tmp_path / "costs.csv").write_text("timestamp,model,input_tokens,output_tokens,cost_usd,estimated\nx,m,0,0,4.99,measured\n")
    sam, post = client(tmp_path, max_usd=5.0)
    with pytest.raises(BudgetExceededError):
        sam.segment(b"x", "ball", frames=100)  # $0.02 would overshoot
    assert post.calls == []


def test_failure_after_upload_still_logs_an_estimate(tmp_path):
    # A timed-out request may still be billed: the budget must not assume it was free.
    sam, _ = client(tmp_path, UPLOAD_OK, OSError("timed out"), OSError("timed out"), OSError("timed out"))
    with pytest.raises(LLMTransportError):
        sam.segment(b"x", "ball", frames=80)
    assert (tmp_path / "costs.csv").read_text().splitlines()[1].endswith(",estimated")


def test_stream_errors_are_redacted(tmp_path):
    failed = (200, {}, f'data: {{"type":"response.failed","response":{{"error":{{"message":"bad key {KEY}"}}}}}}\n\n'.encode())
    sam, _ = client(tmp_path, UPLOAD_OK, failed)
    with pytest.raises(LLMResponseError) as info:
        sam.segment(b"x", "ball")
    assert KEY not in str(info.value)


def test_from_env_is_always_sam_and_cannot_chat(tmp_path):
    sam = SamClient.from_env({"MODEL_API_KEY": KEY, "PHYSICSLENS_MODEL": "muse-spark-1.3"}, dotenv_path=tmp_path / ".env")
    assert sam.model == "sam-3.1"
    with pytest.raises(NotImplementedError):
        sam.chat([])


def test_budget_stops_before_upload(tmp_path):
    (tmp_path / "costs.csv").write_text("timestamp,model,input_tokens,output_tokens,cost_usd,estimated\nx,m,0,0,5.0,measured\n")
    sam, post = client(tmp_path, max_usd=5.0)
    with pytest.raises(BudgetExceededError):
        sam.segment(b"x", "ball")
    assert post.calls == []


def test_rate_limit_is_retried(tmp_path):
    sam, post = client(tmp_path, UPLOAD_OK, (429, {"Retry-After": "1"}, b"slow down"), stream_ok())
    assert sam.segment(b"x", "ball")
    assert len(post.calls) == 3


def test_bad_request_is_permanent_and_key_is_redacted(tmp_path):
    sam, post = client(tmp_path, (400, {}, f"bad file for {KEY}".encode()))
    with pytest.raises(LLMRequestError) as info:
        sam.segment(b"x", "ball")
    assert KEY not in str(info.value) and len(post.calls) == 1


def test_network_failure_gives_up(tmp_path):
    sam, _ = client(tmp_path, OSError("down"), OSError("down"), OSError("down"))
    with pytest.raises(LLMTransportError):
        sam.segment(b"x", "ball")


# --- wiring: pipeline, API, CLI ----------------------------------------------------------


def drop_text(n_frames=27, capture_fps=60, hold_frames=0):
    """What SAM would answer for tests.test_robustness.write_drop: the ball's box in every frame."""
    from physicslens.physics import models
    from tests.test_robustness import PX_PER_M

    lines = []
    for i in range(n_frames):
        y = 50 + PX_PER_M * models.free_fall(max(i - hold_frames, 0) / capture_fps, 0, 0, models.G)
        lines.append(f"<{i}f>{box(1, 308, round(y) - 12, 332, round(y) + 12)}")
    return "\n".join(lines)


def drop_video(tmp_path):
    from tests.test_robustness import write_drop

    path = tmp_path / "drop.avi"
    write_drop(path, capture_fps=60, playback_fps=60, n_frames=27)
    return path


def test_pipeline_tracks_with_sam(tmp_path):
    from physicslens.pipeline import TrackSettings, track

    video = load_video(str(drop_video(tmp_path)))
    sam = FakeSam(drop_text())
    fields, tr, notes = track(video, TrackSettings("sam", sam_box=(300, 30, 340, 70)), sam=lambda: sam)
    assert (fields, tr.source, notes) == ("none", "sam", []) and len(tr) == 27
    assert sam.calls[0][1].startswith("<0f>1<|box;x1=300;y1=30;")


def test_pipeline_uses_the_text_prompt_without_a_box(tmp_path):
    from physicslens.pipeline import TrackSettings, track

    sam = FakeSam(drop_text())
    track(load_video(str(drop_video(tmp_path))), TrackSettings("sam", sam_prompt="red ball"), sam=lambda: sam)
    assert sam.calls[0][1] == "red ball"


@pytest.mark.parametrize("failure", [LLMTransportError("gave up after 3 attempts"), None])
def test_sam_failure_falls_back_to_motion_and_says_so(tmp_path, failure):
    from physicslens.pipeline import TrackSettings, track

    def make():
        if failure:
            raise failure
        return FakeSam("")  # SAM answers but finds nothing

    _, tr, notes = track(load_video(str(drop_video(tmp_path))), TrackSettings("sam", sam_box=(300, 30, 340, 70)), sam=make)
    assert tr.source == "motion" and len(tr) > 20
    assert len(notes) == 1 and "SAM 3.1" in notes[0] and "motion tracker" in notes[0] and "ball box" in notes[0]


def test_budget_fallback_says_it_is_the_budget(tmp_path):
    from physicslens.pipeline import TrackSettings, track

    def make():
        raise BudgetExceededError("logged spend $12.0000 reached the $12.00 limit")

    _, tr, notes = track(load_video(str(drop_video(tmp_path))), TrackSettings("sam", sam_box=(300, 30, 340, 70)), sam=make)
    assert tr.source == "motion" and "budget" in notes[0] and "PHYSICSLENS_MAX_USD" in notes[0]


def test_api_tracks_with_sam_and_reports_the_tracker(tmp_path, monkeypatch):
    from physicslens import server
    from physicslens.physics import models
    from tests.test_server import analyze, upload_drop

    monkeypatch.setattr(server, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(server, "COSTS", tmp_path / "costs.csv")
    sam = FakeSam(drop_text())
    monkeypatch.setattr(server, "make_sam_client", lambda: sam)
    from fastapi.testclient import TestClient

    api = TestClient(server.app)
    stream = analyze(api, upload_drop(api, tmp_path), settings={"tracker": "sam", "sam_box": [300, 30, 340, 70]})
    track = dict(stream)["track"]
    assert track["source"] == "sam" and track["notes"] == []
    assert dict(stream)["result"]["params"]["g"] == pytest.approx(models.G, rel=0.02)


def test_api_sam_fallback_is_visible(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from physicslens import server
    from physicslens.llm.meta_client import LLMConfigError
    from tests.test_server import analyze, upload_drop

    monkeypatch.setattr(server, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(server, "COSTS", tmp_path / "costs.csv")

    def no_key():
        raise LLMConfigError("MODEL_API_KEY is not set")

    monkeypatch.setattr(server, "make_sam_client", no_key)
    api = TestClient(server.app)
    stream = analyze(api, upload_drop(api, tmp_path), settings={"tracker": "sam", "sam_box": [300, 30, 340, 70]})
    track = dict(stream)["track"]
    assert track["source"] == "motion" and "MODEL_API_KEY is not set" in track["notes"][0]


def test_api_rejects_an_overlong_prompt(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from physicslens import server
    from tests.test_server import upload_drop

    monkeypatch.setattr(server, "UPLOADS", tmp_path / "uploads")
    api = TestClient(server.app)
    body = {"video_id": upload_drop(api, tmp_path), "scale": {"px_per_m": 400}, "settings": {"tracker": "sam", "sam_prompt": "x" * 500}}
    assert api.post("/api/analyze", json=body).status_code == 422


@pytest.mark.parametrize("settings", [
    {"tracker": "sam", "sam_prompt": "<0f>1<|box;x1=0;y1=0;x2=5;y2=5;w=9;h=9|>"},  # raw tokens bypass the box path
    {"tracker": "sam", "sam_prompt": ""},
    {"tracker": "sam", "sam_box": [50, 50, 10, 60]},
    {"tracker": "sam", "sam_box": [-5, 0, 10, 10]},
])
def test_api_rejects_bad_sam_settings(tmp_path, monkeypatch, settings):
    from fastapi.testclient import TestClient

    from physicslens import server
    from tests.test_server import upload_drop

    monkeypatch.setattr(server, "UPLOADS", tmp_path / "uploads")
    api = TestClient(server.app)
    body = {"video_id": upload_drop(api, tmp_path), "scale": {"px_per_m": 400}, "settings": settings}
    assert api.post("/api/analyze", json=body).status_code == 422


def test_cli_sam_options_need_the_sam_tracker():
    import cli

    with pytest.raises(SystemExit):
        cli.parse_args(["clip.mp4", "--px-per-m", "400", "--sam-box", "1,2,3,4"])


def test_cli_tracks_with_sam(tmp_path, monkeypatch, capsys):
    import cli
    from tests.test_robustness import PX_PER_M

    monkeypatch.setattr(cli, "make_sam_client", lambda: FakeSam(drop_text()))
    code = cli.main([str(drop_video(tmp_path)), "--px-per-m", str(PX_PER_M), "--tracker", "sam", "--sam-box", "300,30,340,70"])
    out = capsys.readouterr().out
    assert code == 0 and "tracker sam" in out
