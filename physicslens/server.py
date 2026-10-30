"""Local web API for the PhysicsLens UI: upload a video, analyze it, stream the agent live.

Binds to 127.0.0.1 only. The Meta key stays inside MetaClient: the browser sees
tool calls and results, never the key. Run with `physicslens-ui`.
"""

import csv
import dataclasses
import json
import math
import os
import queue
import re
import threading
import uuid
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .agent.loop import run_agent
from .agent.tools import Toolbox, total_stderr
from .calibrate import Calibration
from .llm.meta_client import PROJECT_ROOT, LLMError, MetaClient
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
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
VIDEO_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
               ".avi": "video/x-msvideo", ".mkv": "video/x-matroska"}
_ID = re.compile(r"[0-9a-f]{32}")

app = FastAPI(title="PhysicsLens")
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
async def upload_video(file: UploadFile) -> dict:
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
    return {"video_id": video_id, "name": file.filename, **info}


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
    if req.agent:
        emit("status", {"stage": "agent"})
        question = agent_question(req.height_m, rest_notes)
        try:
            outcome = run_agent(make_client(), box, question,
                                on_message=lambda m: emit("agent_message", dataclasses.asdict(m)))
        except (LLMError, OSError, ValueError) as exc:
            emit("error", {"stage": "agent", "message": str(exc)})
            return
        finally:
            if cost := _cost_event(before):
                emit("cost", cost)
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
    return f"event: {name}\ndata: {json.dumps(_clean(data), allow_nan=False)}\n\n"


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> StreamingResponse:
    _video_path(req.video_id)  # 404 before the stream starts
    if not _busy.acquire(blocking=False):
        raise HTTPException(409, "an analysis is already running")
    events: queue.Queue[str | None] = queue.Queue()

    def work() -> None:
        try:
            _analyze(req, lambda name, data: events.put(_event(name, data)))
        except Exception as exc:  # noqa: BLE001 - the stream must end with an error event, never hang the UI
            events.put(_event("error", {"stage": "internal", "message": f"{type(exc).__name__}: {exc}"}))
        finally:
            events.put(_event("done", {}))
            events.put(None)
            _busy.release()

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (item := events.get()) is not None:
            yield item

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PHYSICSLENS_PORT", "8000")))
