"""Local web API for the PhysicsLens UI: upload a video, analyze it, stream the agent live.

Binds to 127.0.0.1 only. The Meta key stays inside MetaClient: the browser sees
tool calls and results, never the key. Run with `physicslens-ui`.
"""

import csv
import dataclasses
import hmac
import json
import logging
import math
import os
import queue
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .agent.loop import run_agent
from .agent.tools import Toolbox, total_stderr
from .calibrate import Calibration
from .llm.meta_client import PROJECT_ROOT, BudgetExceededError, LLMError, MetaClient
from .llm.sam_client import SamClient
from .physics import models
from .pipeline import (
    MAX_WIDTH,
    TrackSettings,
    agent_question,
    choose,
    fitted_xy,
    load,
    reusable,
    run_fits,
    to_track_data,
    track,
    trim_rest,
    wrong_object,
)
from .track.sam import MAX_TEXT_PROMPT, TEXT_PROMPT_PATTERN

UPLOADS = PROJECT_ROOT / "cache" / "uploads"
WEB_DIST = PROJECT_ROOT / "web" / "dist"
COSTS = PROJECT_ROOT / "costs.csv"
RUNS = PROJECT_ROOT / "cache" / "runs"  # one JSON file per analysis: the History page
# Running as a public site (a Hugging Face Space) sets these; locally nothing changes.
PUBLIC = os.environ.get("PHYSICSLENS_PUBLIC") == "1"  # the UI says that videos and runs are visible to everyone
MAX_UPLOAD_BYTES = int(os.environ.get("PHYSICSLENS_MAX_UPLOAD_MB", "500")) * 1024 * 1024
MAX_UPLOADS = int(os.environ.get("PHYSICSLENS_MAX_UPLOADS", "0"))  # newest videos kept; 0: all of them
KEEPALIVE_S = 15.0  # a tunnel or proxy closes a response silent for ~100 s
# A public site: deleting runs needs this token (unset: nobody deletes), and each visitor gets so many per hour.
ADMIN_TOKEN = os.environ.get("PHYSICSLENS_ADMIN_TOKEN", "")
RATE_ANALYZE = int(os.environ.get("PHYSICSLENS_RATE_ANALYZE", "6"))
RATE_UPLOAD = int(os.environ.get("PHYSICSLENS_RATE_UPLOAD", "12"))
RATE_WINDOW_S = 3600.0
_hits: dict[tuple[str, str], list[float]] = {}
_hits_lock = threading.Lock()
_cancels: dict[str, threading.Event] = {}  # runs in progress: set to stop one


class Cancelled(Exception):
    """The visitor stopped the run or left; raised at the next event so no further paid step starts."""


def _visitor(request: Request) -> str:
    # Behind the Cloudflare tunnel uvicorn already gives the real client (it trusts X-Forwarded-For from loopback).
    # If a request still shows loopback, Cloudflare's CF-Connecting-IP names the client; anyone else could write
    # that header, so it is trusted only from loopback.
    host = request.client.host if request.client else "unknown"
    if host in ("127.0.0.1", "::1", "testclient"):
        return request.headers.get("cf-connecting-ip") or host
    return host


def _limit(request: Request, kind: str, per_hour: int) -> None:
    """On a public site, refuse a visitor's request beyond `per_hour` of this kind in the last hour."""
    if not PUBLIC or per_hour <= 0:
        return
    key, now = (_visitor(request), kind), time.monotonic()
    with _hits_lock:
        if len(_hits) > 10_000:  # forget visitors idle for an hour, so many addresses cannot grow memory
            for old in [k for k, ts in _hits.items() if not ts or now - ts[-1] >= RATE_WINDOW_S]:
                del _hits[old]
        recent = [t for t in _hits.get(key, []) if now - t < RATE_WINDOW_S]
        if len(recent) >= per_hour:
            wait = int((RATE_WINDOW_S - (now - recent[0])) / 60) + 1
            raise HTTPException(429, f"{per_hour} {kind}s per hour from your network on this public demo: try again in {wait} min")
        _hits[key] = [*recent, now]


def _refund(request: Request, kind: str) -> None:
    """Give back the last counted request (it was refused for another reason, e.g. the slot was busy)."""
    with _hits_lock:
        recent = _hits.get((_visitor(request), kind))
        if recent:
            recent.pop()
VIDEO_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
               ".avi": "video/x-msvideo", ".mkv": "video/x-matroska"}
_ID = re.compile(r"[0-9a-f]{32}")
MAX_NAME = 200  # characters of an uploaded file name kept for History
logger = logging.getLogger("physicslens.server")

app = FastAPI(title="PhysicsLens")


@app.middleware("http")
async def no_index(request, call_next):
    """The API serves personal runs and videos: never indexed, whatever reaches it."""
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response
_busy = threading.Lock()  # one analysis at a time: frames are memory-heavy and the budget is shared


def make_client():
    """The runtime model; tests swap this for a scripted MockClient."""
    return MetaClient.from_env()


def make_sam_client():
    """SAM 3.1 for the "sam" tracker; tests swap this for a scripted fake."""
    return SamClient.from_env(cost_log=COSTS)


def _video_path(video_id: str) -> Path:
    # The id is generated here on upload; anything else is refused before touching the filesystem.
    if not _ID.fullmatch(video_id):
        raise HTTPException(404, "unknown video")
    matches = [p for p in UPLOADS.glob(f"{video_id}.*") if p.suffix in VIDEO_TYPES]
    if not matches:
        raise HTTPException(404, "unknown video")
    return matches[0]


def _probe(path: Path) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if not width or not height:
        return None
    return {"fps": fps, "frames": frames, "width": width, "height": height,
            "duration_s": frames / fps if fps > 0 else None}


@app.post("/api/videos")
async def upload_video(file: UploadFile, request: Request) -> dict:
    _limit(request, "upload", RATE_UPLOAD)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_TYPES:
        raise HTTPException(415, f"unsupported video type {ext or '(none)'}; use {', '.join(sorted(VIDEO_TYPES))}")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    video_id = uuid.uuid4().hex
    dest = UPLOADS / f"{video_id}{ext}"
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"video larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            out.write(chunk)
    info = _probe(dest)
    if info is None:
        dest.unlink(missing_ok=True)
        raise HTTPException(415, "OpenCV cannot decode this video")
    name = (file.filename or "")[:MAX_NAME]
    try:  # the name is kept beside the video for History; the id stays the only thing in any path
        (UPLOADS / f"{video_id}.json").write_text(json.dumps({"name": name}), encoding="utf-8")
    except OSError:
        logger.warning("video %s: name not saved", video_id)
    _prune_uploads()
    return {"video_id": video_id, "name": name, **info}


def _prune_uploads() -> None:
    """Keep the newest MAX_UPLOADS videos so a public disk never fills; runs of a removed video show it as gone."""
    if MAX_UPLOADS <= 0:
        return
    videos = sorted((p for p in UPLOADS.iterdir() if p.suffix in VIDEO_TYPES and _ID.fullmatch(p.stem)), key=lambda p: p.stat().st_mtime)
    for old in videos[: max(0, len(videos) - MAX_UPLOADS)]:
        old.unlink(missing_ok=True)
        (UPLOADS / f"{old.stem}.json").unlink(missing_ok=True)


@app.get("/api/config")
def config() -> dict:
    return {"public": PUBLIC, "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024), "can_delete": not PUBLIC}


@app.get("/api/videos/{video_id}/info")
def video_info(video_id: str) -> dict:
    """What the stage needs to show a saved run's video again."""
    info = _probe(_video_path(video_id))
    if info is None:
        raise HTTPException(415, "OpenCV cannot decode this video")
    return {"video_id": video_id, "name": _video_name(video_id) or "", **info}


@app.get("/api/videos/{video_id}/file")
def video_file(video_id: str) -> FileResponse:
    path = _video_path(video_id)
    return FileResponse(path, media_type=VIDEO_TYPES[path.suffix])


@app.get("/api/videos/{video_id}/frame")
def video_frame(video_id: str, t: float = 0.0) -> Response:
    cap = cv2.VideoCapture(str(_video_path(video_id)))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0.0) * 1000)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise HTTPException(416, f"no frame at {t} s")
    return Response(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(), media_type="image/jpeg")


class Scale(BaseModel):
    """Either px_per_m, or two points on a reference object (source pixels) and its length."""

    px_per_m: float | None = Field(None, gt=0)
    p1: tuple[float, float] | None = None
    p2: tuple[float, float] | None = None
    length_m: float | None = Field(None, gt=0)

    def calibration(self) -> Calibration:
        if self.px_per_m:
            return Calibration.from_reference(self.px_per_m, 1.0)
        if self.p1 and self.p2 and self.length_m:
            return Calibration.from_points(self.p1, self.p2, self.length_m)
        raise ValueError("scale needs px_per_m, or p1, p2 and length_m")


Coord = Annotated[int, Field(ge=0, le=20000)]


class Settings(BaseModel):
    """Tracking settings in source pixels, as the user sees the video."""

    tracker: Literal["motion", "color", "sam"] = "motion"
    hsv_lower: tuple[int, int, int] = (0, 0, 200)
    hsv_upper: tuple[int, int, int] = (180, 60, 255)
    fields: Literal["none", "auto", "top", "bottom"] = "none"
    roi: tuple[int, int, int, int] | None = None
    min_area: int = Field(10, ge=1)
    max_area: int | None = Field(None, ge=1)
    max_jump: float | None = Field(None, gt=0)
    # Around the object on the first frame of the window, source pixels.
    sam_box: tuple[Coord, Coord, Coord, Coord] | None = None
    # Opt-in: text misses small balls.
    sam_prompt: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_PROMPT, pattern=TEXT_PROMPT_PATTERN)

    @field_validator("sam_box")
    @classmethod
    def _ordered(cls, box):
        if box is not None and not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError("sam_box must be x0,y0,x1,y1 with x0 < x1 and y0 < y1")
        return box

    def scaled(self, s: float, px_per_m: float | None = None) -> TrackSettings:
        """The same settings in decoded pixels, after the frames were shrunk by `s`; `px_per_m` in source pixels."""
        return TrackSettings(
            tracker=self.tracker, hsv_lower=self.hsv_lower, hsv_upper=self.hsv_upper, fields=self.fields, roi=self.roi,
            min_area=self.min_area, max_area=self.max_area, max_jump=self.max_jump, sam_box=self.sam_box,
            sam_prompt=self.sam_prompt, px_per_m=px_per_m,
        ).scaled(s)


class AnalyzeRequest(BaseModel):
    video_id: str
    scale: Scale
    settings: Settings = Settings()
    start: float = Field(0.0, ge=0)
    end: float | None = Field(None, gt=0)
    capture_fps: float | None = Field(None, gt=0)
    model: Literal["auto", "free_fall", "projectile", "pendulum"] = "auto"
    axis: Literal["x", "y"] = "x"
    agent: bool = False
    height_m: float | None = Field(None, gt=0)
    detect_release: bool = True  # drop samples where the ball is still before release / after landing


def _clean(value):
    """JSON-safe: NaN and inf become null, numpy scalars become floats."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _spent() -> float | None:
    """Logged spend; None when the log is unreadable, so cost reporting never hides the real error."""
    if not COSTS.is_file():
        return 0.0
    try:
        with COSTS.open(newline="") as f:
            return sum(float(row["cost_usd"]) for row in csv.DictReader(f))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _cost_event(before: float | None) -> dict | None:
    now = _spent()
    if before is None or now is None:
        return None
    return {"run_usd": now - before, "total_usd": now}


def _result(box: Toolbox, fit_id: str, px_per_m: float, t_offset: float, time_scale: float, check: dict) -> dict:
    fit = box.fits[fit_id]
    curve = fitted_xy(box, fit_id)
    _, model, axis, _ = box.windows[fit_id]
    result = {
        "fit_id": fit_id,
        "model": model,
        "axis": axis,
        "params": fit.params,
        "stderr": fit.stderr,
        # Fit noise plus the scale reading (READ_PX on the reference); perspective is not in it, the checks flag it.
        "stderr_total": total_stderr(fit, box.scale_rel_err),
        "scale_rel_err": box.scale_rel_err,
        "deviation_pct": {"g": 100 * (fit.params["g"] - models.G) / models.G} if "g" in fit.params else {},
        "r2": fit.r2,
        "rmse_m": fit.rmse,
        "check": check,
        "curve": {
            "t_play": [(t_offset + t) / time_scale for t in curve["t"]],
            "x_px": [x * px_per_m for x in curve["x"]],
            "y_px": [y * px_per_m for y in curve["y"]],
        },
    }
    if model == "pendulum":
        result["length_m"] = models.pendulum_length(fit.params["omega"], fit.params["gamma"])
        result["period_s"] = 2 * math.pi / fit.params["omega"]
    return result


def _analyze(req: AnalyzeRequest, emit) -> None:
    path = _video_path(req.video_id)
    emit("status", {"stage": "loading"})
    try:
        calibration = req.scale.calibration()
        video, video_warnings = load(str(path), req.start, req.end, req.capture_fps, MAX_WIDTH)
    except (ValueError, FileNotFoundError) as exc:
        emit("error", {"stage": "loading", "message": str(exc)})
        return
    s = video.scale
    emit("video", {"frames": len(video.frames), "fps": video.fps, "timestamp_source": video.timestamp_source,
                   "scale": s, "warnings": video_warnings})

    before = _spent()
    emit("status", {"stage": "tracking"})
    try:
        fields, tr, track_notes = track(
            video, req.settings.scaled(s, calibration.px_per_m), sam=lambda: make_sam_client(),
            on_detect=lambda d: emit("detection", {"box": [v / s for v in d.box], "frame": d.frame, "t": float(video.timestamps[d.frame]),
                                                  "model": d.model, "r2": d.r2}),
        )
    except ValueError as exc:
        emit("error", {"stage": "tracking", "message": str(exc)})
        return
    finally:
        if req.settings.tracker == "sam" and (cost := _cost_event(before)):
            emit("cost", cost)
    full = tr
    rest_notes: list[str] = []
    rest = np.zeros(len(full), bool)  # dropped samples, still drawn on the video so the user sees them
    if req.detect_release:
        trim = trim_rest(tr, calibration.px_per_m * s)
        tr, rest_notes = trim.track, trim.notes
        rest[:trim.lead] = True
        rest[len(full) - trim.trail:] = True
    # Decoded pixels per metre: the frames were shrunk by s after the user calibrated on the source.
    data = to_track_data(tr, calibration.scaled(s))
    box = Toolbox(data, scale_rel_err=calibration.rel_err)
    t_offset = req.start + float(tr.t[0])  # real seconds from the clip start to the first sample
    emit("track", {
        "fields": fields,
        "source": tr.source,
        "px_per_m": calibration.px_per_m,
        "t": data.t.tolist(),
        "t_play": [(req.start + t) / video.time_scale for t in tr.t],
        "x_px": (tr.x / s).tolist(),
        "y_px": (tr.y / s).tolist(),
        "x_m": data.x.tolist(),
        "y_m": data.y.tolist(),
        "notes": [*track_notes, *rest_notes],
        "rest": {
            "t_play": [(req.start + t) / video.time_scale for t in full.t[rest]],
            "x_px": (full.x[rest] / s).tolist(),
            "y_px": (full.y[rest] / s).tolist(),
        },
    })

    if req.agent and (why := wrong_object(data, calibration.rel_err)):
        emit("error", {"stage": "no measurement", "message": why, "steps": 0})
        return
    outcome = None
    if req.agent:
        emit("status", {"stage": "agent"})
        question = agent_question(req.height_m, rest_notes)
        try:
            outcome = run_agent(make_client(), box, question,
                                on_message=lambda m: emit("agent_message", dataclasses.asdict(m)))
        except BudgetExceededError as exc:
            # The demo never breaks on the budget: the deterministic fit and checks still give a number.
            emit("note", {"message": f"The Muse agent was skipped: the API budget is used up ({exc}). Fitted without the agent."})
        except (LLMError, OSError, ValueError) as exc:
            emit("error", {"stage": "agent", "message": str(exc)})
            return
        finally:
            if cost := _cost_event(before):
                emit("cost", cost)
    if outcome is not None:
        if outcome.fit_id is None or outcome.check is None or outcome.fit_id not in box.fits:
            stage = "no measurement" if outcome.no_measurement else "agent"
            emit("error", {"stage": stage, "message": outcome.summary, "steps": outcome.steps})
            return
        fit_id, check = outcome.fit_id, outcome.check
        extra = {"summary": outcome.summary, "steps": outcome.steps, "source": "agent"}
    else:
        emit("status", {"stage": "fitting"})
        results = run_fits(box, req.model, req.axis)
        emit("fits", {"fits": results})
        if why := wrong_object(data, calibration.rel_err, reusable(results, req.model, req.axis)):
            emit("error", {"stage": "no measurement", "message": why, "steps": 0})
            return
        best = choose(results)
        if best is None:
            emit("error", {"stage": "fitting", "message": "no model could be fitted"})
            return
        fit_id, check = best["fit_id"], best["check"]
        extra = {"source": "deterministic"}

    result = {**_result(box, fit_id, calibration.px_per_m, t_offset, video.time_scale, check), **extra}
    if req.height_m and result["model"] == "projectile":
        landing = box.call("predict_landing", {"fit_id": fit_id, "height_m": req.height_m})
        # A failed prediction is said, never shown as a landing point (the UI would read missing numbers).
        result["landing" if "error" not in landing else "landing_error"] = landing.get("error", landing)
    emit("result", result)


def _event(name: str, data) -> str:
    return f"event: {name}\ndata: {json.dumps(data, allow_nan=False)}\n\n"


def _video_name(video_id: str) -> str | None:
    try:
        return json.loads((UPLOADS / f"{video_id}.json").read_text()).get("name")
    except (OSError, ValueError, AttributeError):
        return None


_MODEL_KEY = {"free_fall": ("g", "m/s²"), "projectile": ("g", "m/s²")}


def summarize(events: list[dict], request: dict) -> dict:
    """What a History row shows, from the run's events: status, the key measured value, cost."""
    last = {e["event"]: e["data"] for e in events if isinstance(e.get("data"), dict)}
    result, error, tr = last.get("result"), last.get("error"), last.get("track")
    key = None
    if result and result.get("model") == "pendulum":
        key = {"name": "L", "value": result.get("length_m"), "err": None, "unit": "m"}
    elif result and (result.get("params") or {}).get("g") is not None:
        name, unit = _MODEL_KEY.get(result.get("model"), ("g", "m/s²"))
        key = {"name": name, "value": result["params"]["g"], "err": (result.get("stderr_total") or {}).get("g"), "unit": unit}
    source = (tr or {}).get("source") or (request.get("settings") or {}).get("tracker") or "motion"
    return {
        "status": ("ok" if (result.get("check") or {}).get("ok") else "flagged") if result else "none",
        "model": result.get("model") if result else None,
        "key": key,
        "dev_pct": (result.get("deviation_pct") or {}).get("g") if result else None,
        "tracker": source.split("-")[0],  # motion-top is the motion tracker on one field
        "agent": bool(request.get("agent")),
        "steps": (result or error or {}).get("steps"),
        "cost_usd": last["cost"].get("run_usd") if "cost" in last else None,
        "error": None if result or not error else {"stage": error.get("stage"), "message": error.get("message")},
        "thumb_t": float(request.get("start") or 0.0),
    }


def _save_run(run_id: str, req: AnalyzeRequest, events: list[dict]) -> None:
    request = req.model_dump(mode="json")
    record = {"id": run_id, "created_at": datetime.now(UTC).isoformat(), "video_id": req.video_id,
              "video_name": _video_name(req.video_id), "request": request, "events": events}
    record["summary"] = {k: record[k] for k in ("id", "created_at", "video_id", "video_name")} | summarize(events, request)
    RUNS.mkdir(parents=True, exist_ok=True)
    # The summary also goes in its own small file: the list never parses the full records.
    _write_private(RUNS / f"{run_id}.json", record)
    _write_private(RUNS / f"{run_id}.sum.json", record["summary"])


def _write_private(path: Path, data) -> None:
    """Atomic and owner-only: prompts, model output, costs and file names stay on this account."""
    tmp = path.with_suffix(".tmp")
    try:
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
            json.dump(data, f, allow_nan=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _run_path(run_id: str) -> Path:
    path = RUNS / f"{run_id}.json"
    if not _ID.fullmatch(run_id) or not path.is_file():
        raise HTTPException(404, "unknown run")
    return path


@app.get("/api/runs")
def list_runs() -> list[dict]:
    summaries = []
    for path in RUNS.glob("*.sum.json") if RUNS.is_dir() else []:
        if not _ID.fullmatch(path.name.removesuffix(".sum.json")):
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # a broken file never hides the other runs
        if isinstance(summary, dict) and isinstance(summary.get("created_at"), str) and isinstance(summary.get("id"), str):
            summaries.append(summary)
    return sorted(summaries, key=lambda r: (r["created_at"], r["id"]), reverse=True)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        return json.loads(_run_path(run_id).read_text())
    except (OSError, ValueError) as exc:
        raise HTTPException(404, "unreadable run") from exc


@app.delete("/api/runs/{run_id}", status_code=204)
def delete_run(run_id: str, request: Request) -> Response:
    # A public History is shared: only whoever runs the site (with the admin token) removes runs from it.
    given = request.headers.get("x-admin-token", "").encode()
    if PUBLIC and not (ADMIN_TOKEN and hmac.compare_digest(given, ADMIN_TOKEN.encode())):
        raise HTTPException(403, "runs on this public demo can only be deleted by its owner")
    path = _run_path(run_id)
    (RUNS / f"{run_id}.sum.json").unlink(missing_ok=True)
    path.unlink(missing_ok=True)  # the video stays: other runs may use it; a second delete is a 404 above
    return Response(status_code=204)


@app.post("/api/runs/{run_id}/cancel", status_code=202)
def cancel_run(run_id: str) -> Response:
    """Stop a run in progress (the Stop button); the slot frees as soon as the current step ends."""
    event = _cancels.get(run_id) if _ID.fullmatch(run_id) else None
    if event is None:
        raise HTTPException(404, "no such run in progress")
    event.set()
    return Response(status_code=202)


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, request: Request) -> StreamingResponse:
    _video_path(req.video_id)  # 404 before the stream starts
    _limit(request, "analysis", RATE_ANALYZE)
    if not _busy.acquire(blocking=False):
        _refund(request, "analysis")  # a busy slot is not the visitor's fault
        raise HTTPException(409, "an analysis is already running")
    events: queue.Queue[str | None] = queue.Queue()
    run_id = uuid.uuid4().hex
    recorded: list[dict] = []
    cancel = _cancels[run_id] = threading.Event()

    def emit(name: str, data) -> None:
        # Once stopped, only what was already paid for (cost) and why it ended (error) still go out.
        if cancel.is_set() and name not in ("cost", "error"):
            raise Cancelled
        clean = _clean(data)
        recorded.append({"event": name, "data": clean})
        events.put(_event(name, clean))

    def work() -> None:
        try:
            emit("run", {"id": run_id})
            _analyze(req, emit)
        except Cancelled:
            stopped = {"stage": "cancelled", "message": "stopped before it finished"}
            recorded.append({"event": "error", "data": stopped})
            events.put(_event("error", stopped))
        except Exception as exc:  # noqa: BLE001 - the stream must end with an error event, never hang the UI
            emit("error", {"stage": "internal", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            try:
                # Saved before "done" reaches the client, so a History refresh on "done" finds the run.
                recorded.append({"event": "done", "data": {}})
                _save_run(run_id, req, recorded)
            except Exception:  # a failed save must never keep the stream open or the lock held
                logger.exception("run %s not saved to History", run_id)
            finally:
                events.put(_event("done", {}))
                events.put(None)
                _cancels.pop(run_id, None)
                _busy.release()

    threading.Thread(target=work, daemon=True).start()

    def stream():
        try:
            while True:
                try:
                    item = events.get(timeout=KEEPALIVE_S)
                except queue.Empty:
                    yield ": keepalive\n\n"  # an SSE comment: proxies see traffic during a long silent step
                    continue
                if item is None:
                    return
                yield item
        finally:
            cancel.set()  # the visitor left (or the run ended): nothing more is paid for an empty room

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


def main() -> None:
    import uvicorn

    # 127.0.0.1 unless told otherwise: only a container (the public Space) listens on every interface.
    uvicorn.run(app, host=os.environ.get("PHYSICSLENS_HOST", "127.0.0.1"), port=int(os.environ.get("PHYSICSLENS_PORT") or os.environ.get("PORT") or "8000"))
