import pytest


@pytest.fixture(autouse=True)
def _history_in_tmp(tmp_path, monkeypatch):
    """Every API test saves its runs under tmp_path: no test ever writes the real History."""
    from physicslens import server

    monkeypatch.setattr(server, "RUNS", tmp_path / "runs")
