"""SAM 3.1 tracker offline: recorded streams and a fake transport stand in for the Meta Model API."""

import json
from pathlib import Path

import numpy as np
import pytest

from physicslens.ingest import Video, load_video
from physicslens.llm.meta_client import BudgetExceededError, LLMRequestError, LLMResponseError, LLMTransportError
from physicslens.llm.sam_client import SamClient, stream_text
from physicslens.track.sam import box_prompt, encode_clip, parse_frames, pick_object, track_sam

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


def test_pick_with_nothing_inside_the_area_is_none():
    assert pick_object({0: {1: (0, 0, 10, 10)}}, roi=(300, 300, 400, 400)) is None


# --- tracking ------------------------------------------------------------------------


def synthetic_video(n=12, fps=30.0) -> Video:
    frames = [np.zeros((480, 640, 3), np.uint8) for _ in range(n)]
    return Video(frames, np.arange(n) / fps, fps, 640, 480, "index", 1.0)


class FakeSam:
    def __init__(self, text):
        self.text, self.calls = text, []

    def segment(self, clip: bytes, prompt: str, direction: str = "both") -> str:
        self.calls.append((clip, prompt, direction))
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


def test_track_sam_ignores_frames_past_the_clip():
    sam = FakeSam("\n".join(f"<{n}f>{box(1, 0, n, 10, n + 10)}" for n in range(15)))
    assert len(track_sam(synthetic_video(), sam, prompt="ball")) == 12


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
    defaults = dict(cache_dir=tmp_path / "cache", cost_log=tmp_path / "costs.csv", post=post, sleep=lambda s: None)
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


def test_empty_result_is_cached_too(tmp_path):
    empty = (200, {}, (FIXTURES / "no_match.sse").read_bytes())
    sam, post = client(tmp_path, UPLOAD_OK, empty)
    assert sam.segment(b"x", "ball") == "" and sam.segment(b"x", "ball") == ""
    assert len(post.calls) == 2


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
