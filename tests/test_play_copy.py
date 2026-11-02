"""A video the browser cannot play (MJPEG, MPEG-4 Part 2...) is streamed as a playable copy; analysis keeps the original."""

import os
import time

import cv2

from physicslens import server
from tests.test_server import client, upload_drop  # noqa: F401  (client is a fixture)


def test_an_unplayable_upload_is_streamed_as_a_playable_copy(client, tmp_path):  # noqa: F811
    video = upload_drop(client, tmp_path)  # MJPEG in AVI: OpenCV reads it, no browser plays it
    original = server._video_path(video)
    assert server._codec(original) == "mjpg"

    response = client.get(f"/api/videos/{video}/file")
    assert response.status_code == 200
    assert response.headers["content-type"] in ("video/mp4", "video/webm")
    copy = server._play_copy(video)
    assert copy is not None and server._codec(copy) in server.PLAYABLE_CODECS
    assert response.content == copy.read_bytes()

    cap = cv2.VideoCapture(str(copy))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 27 and cap.get(cv2.CAP_PROP_FPS) == 60  # same clock as the track
    cap.release()
    assert server._video_path(video) == original  # the analysis never reads the copy

    mtime = os.stat(copy).st_mtime_ns
    assert client.get(f"/api/videos/{video}/file").status_code == 200
    assert os.stat(copy).st_mtime_ns == mtime  # made once


def test_a_playable_upload_is_streamed_as_is(client, tmp_path):  # noqa: F811
    video = upload_drop(client, tmp_path)
    copy = server._write_play_copy(server._video_path(video), video)  # an H.264/VP8 clip, uploaded as such
    with copy.open("rb") as f:
        body = client.post("/api/videos", files={"file": ("phone" + copy.suffix, f, "video/mp4")}).json()
    assert client.get(f"/api/videos/{body['video_id']}/file").content == copy.read_bytes()
    assert server._play_copy(body["video_id"]) is None


def test_pruning_a_video_removes_its_copy(client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "MAX_UPLOADS", 1)
    (tmp_path / "a").mkdir()
    first = upload_drop(client, tmp_path / "a")
    client.get(f"/api/videos/{first}/file")
    assert server._play_copy(first) is not None
    time.sleep(0.01)
    (tmp_path / "b").mkdir()
    upload_drop(client, tmp_path / "b")
    assert server._play_copy(first) is None
