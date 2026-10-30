"""Settings for running PhysicsLens as a public site: the UI is told it is public, and old uploads make room."""

import os
import time

from physicslens import server
from tests.test_server import client, upload_drop  # noqa: F401  (client is a fixture)


def test_the_ui_is_told_whether_the_site_is_public(client, monkeypatch):  # noqa: F811
    assert client.get("/api/config").json() == {"public": False, "max_upload_mb": 500}
    monkeypatch.setattr(server, "PUBLIC", True)
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 100 * 1024 * 1024)
    assert client.get("/api/config").json() == {"public": True, "max_upload_mb": 100}


def test_only_the_newest_uploads_are_kept(client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "MAX_UPLOADS", 2)
    ids = []
    for i in range(3):
        (tmp_path / str(i)).mkdir()
        ids.append(upload_drop(client, tmp_path / str(i)))
        time.sleep(0.01)  # distinct modification times
    assert client.get(f"/api/videos/{ids[0]}/file").status_code == 404  # the oldest made room
    assert not (server.UPLOADS / f"{ids[0]}.json").exists()  # with its name file
    assert all(client.get(f"/api/videos/{i}/file").status_code == 200 for i in ids[1:])


def test_no_limit_by_default_keeps_every_upload(client, tmp_path):  # noqa: F811
    for i in range(3):
        (tmp_path / str(i)).mkdir()
        upload_drop(client, tmp_path / str(i))
    assert len([p for p in server.UPLOADS.iterdir() if p.suffix == ".avi"]) == 3


def test_settings_come_from_the_environment(monkeypatch):
    import importlib

    monkeypatch.setenv("PHYSICSLENS_PUBLIC", "1")
    monkeypatch.setenv("PHYSICSLENS_MAX_UPLOAD_MB", "100")
    monkeypatch.setenv("PHYSICSLENS_MAX_UPLOADS", "40")
    try:
        reloaded = importlib.reload(server)
        assert reloaded.PUBLIC is True and reloaded.MAX_UPLOAD_BYTES == 100 * 1024 * 1024 and reloaded.MAX_UPLOADS == 40
    finally:
        for name in ("PHYSICSLENS_PUBLIC", "PHYSICSLENS_MAX_UPLOAD_MB", "PHYSICSLENS_MAX_UPLOADS"):
            os.environ.pop(name, None)
        importlib.reload(server)
