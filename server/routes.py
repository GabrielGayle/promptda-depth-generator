"""Infer + run + evaluate API routes."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import imageio
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from depth_core.infer_backend import DEFAULT_INFER_BACKEND, modal_ready, resolve_backend
from depth_core.modal_pricing import pricing_meta
from depth_core.pipeline import (
    BLEND_MODES,
    DEFAULT_MODEL,
    LOOK_PRESETS,
    MAX_SIZE,
    MODEL_CHOICES,
    OUTPUT_DIR,
    SCENERY_MATTE_MODES,
    depth_percentile_at,
    get_backend,
    list_recent_runs,
    load_recipe,
    normalize_model_key,
    run_pair,
    run_rgb,
    run_zip,
    save_recipe,
)
from depth_core.batch_stills import (
    BATCH_MAX_FOLDERS,
    batch_info,
    export_folder as export_batch_folder,
    discover_batches,
    folder_summary,
    images_in_folder,
    prepare_batch,
    run_batch,
    run_folders,
    run_preview as run_batch_preview,
)
from depth_core.film import (
    FILM_LIVE,
    clip_info,
    export_mp4 as export_film_mp4,
    film_meta,
    load_params as load_film_params,
    normalize_params as normalize_film_params,
    prepare_film,
    render_vis as render_film_vis,
    run_full as run_film_full,
    run_preview as run_film_preview,
    sample_depth as sample_film_depth,
    save_params as save_film_params,
)
from depth_core.video import (
    export_frames_to_mp4,
    infer_prepared_run,
    load_frames_meta,
    prepare_video,
    run_video,
)
from server.arena import (
    ArenaError,
    auth_status,
    download_block_image,
    list_recent_channels,
    publish_image_to_channel,
    resolve_ref,
)
from server.arena_export import build_arena_publish_fields, scan_arena_kitchens
from server.batch import start_arena_roll_job
from server.cache import get_run, put_run, set_frame, update_run
from server.graph_eval import (
    DEFAULT_RELIEF_GC_ALPHA,
    DEFAULT_RELIEF_INVERT,
    DEFAULT_RELIEF_MAX_SIDE,
    RELIEF_MAX_SIDE_CHOICES,
    clear_default_template,
    default_graph,
    evaluate_graph,
    load_default_template,
    render_tune_turbo_preview,
    save_default_template,
    template_to_graph,
)
from server.modal_usage import get_modal_usage, reset_modal_usage
from server.operator_eval import (
    clear_default_operator_patch,
    evaluate_operator,
    export_operator_video,
    get_operator_default,
    operator_stream_jpeg,
    save_default_operator_patch,
)
from server.status import (
    build_status_async,
    record_ui_heartbeat,
    set_local_infer,
    set_modal_state,
    set_warmup,
)
from depth_core.operator import operator_meta, normalize_patch
from depth_core.temporal import freq_fuse_meta, smooth_meta

router = APIRouter(prefix="/api")

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


class GraphBody(BaseModel):
    version: int = 2
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


class EvaluateBody(BaseModel):
    graph: GraphBody


class FrameBody(BaseModel):
    index: int
    graph: Optional[GraphBody] = None


class ExportBody(BaseModel):
    graph: GraphBody


class OperatorPatchBody(BaseModel):
    version: int = 1
    algorithm: int = 1
    oscillators: dict[str, Any] = Field(default_factory=dict)
    filter: dict[str, Any] = Field(default_factory=dict)
    shaper: dict[str, Any] = Field(default_factory=dict)
    envelope: dict[str, Any] = Field(default_factory=dict)
    lfo: dict[str, Any] = Field(default_factory=dict)
    global_: dict[str, Any] = Field(default_factory=dict, alias="global")
    t: Optional[float] = None

    model_config = {"populate_by_name": True}

    def to_patch(self) -> dict[str, Any]:
        data = self.model_dump(by_alias=True)
        data.pop("global_", None)
        return normalize_patch(data)


class ArenaResolveBody(BaseModel):
    url: str
    mode: str = "auto"  # auto | shuffle


class ArenaInferBody(BaseModel):
    url: str
    mode: str = "auto"  # auto | shuffle — shuffle picks a random imageable block
    model_key: str = DEFAULT_MODEL
    max_size: int = 756
    infer_backend: Optional[str] = None


class ArenaPublishBody(BaseModel):
    run_id: str
    channel_id: str
    title: Optional[str] = None
    filename: str = "blend_live.png"


class ArenaKitchensBody(BaseModel):
    url: str
    limit: int = 40


class ArenaRollBody(BaseModel):
    arena_url: str
    count: int = 5
    model_key: str = DEFAULT_MODEL
    max_size: int = 756
    infer_backend: Optional[str] = "modal"
    wackiness: float = 0.45
    max_fidelity: bool = False
    channel_id: str
    title: str = ""
    graph: GraphBody


def _file_url(run_id: str, name: str) -> str:
    return f"/api/runs/{run_id}/files/{name}"


def _begin_infer(infer_backend: Optional[str]) -> str:
    backend = resolve_backend(infer_backend)
    if backend == "modal":
        set_modal_state("ready")
    else:
        set_local_infer("running")
    return backend


def _end_infer(
    backend: str,
    state: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    run_id = None
    device = None
    if state:
        run_id = Path(state.get("out_dir") or "").name or state.get("run_id")
        device = state.get("infer_device")
    if error:
        if backend == "modal":
            set_modal_state("error", last_job=run_id, last_error=error)
        else:
            set_local_infer("failed", last_run_id=run_id, last_error=error, device=device)
        return
    if backend == "modal":
        set_modal_state("ready", last_job=run_id)
    else:
        set_local_infer("idle", last_run_id=run_id, device=device)


def _infer_response(state: dict, *, evaluate: bool = True) -> dict:
    run_id = put_run(state)
    backend = str(state.get("infer_backend") or "local")
    if backend == "modal":
        set_modal_state("ready", last_job=run_id)
    else:
        set_local_infer(
            "idle",
            last_run_id=run_id,
            device=state.get("infer_device"),
        )
    kind = state.get("kind") or "still"
    if int(state.get("frame_count") or 1) > 1:
        kind = "video"
    graph = default_graph(run_id, kind=kind)
    if evaluate and state.get("depth") is not None:
        try:
            evaluate_graph(run_id, graph)
        except Exception:
            pass
    look_urls = {
        k: _file_url(run_id, Path(p).name) for k, p in state.get("look_paths", {}).items()
    }
    source_file = state.get("source_file")
    preview_gif = state.get("preview_gif")
    if not preview_gif or not source_file or state.get("frame_width") is None:
        meta = load_frames_meta(Path(state["out_dir"]))
        if meta:
            preview_gif = preview_gif or meta.get("preview_gif")
            source_file = source_file or meta.get("source_file")
            if state.get("frame_width") is None:
                state["frame_width"] = meta.get("frame_width")
                state["frame_height"] = meta.get("frame_height")
            if state.get("ready_indices") is None:
                state["ready_indices"] = meta.get("ready_indices")
            if state.get("infer_mode") is None:
                state["infer_mode"] = meta.get("infer_mode")
    out = {
        "run_id": run_id,
        "kind": state.get("kind") or "still",
        "frame_count": int(state.get("frame_count") or 1),
        "frame_index": int(state.get("frame_index") or 0),
        "ready_count": int(state.get("ready_count") or 0)
        if str(state.get("infer_status") or "") in ("pending", "running")
        else int(state.get("ready_count") or state.get("frame_count") or 1),
        "infer_status": state.get("infer_status") or "done",
        "sample_fps": float(state.get("sample_fps") or 10),
        "model_key": state.get("model_key"),
        "max_size": state.get("max_size"),
        "frame_width": state.get("frame_width"),
        "frame_height": state.get("frame_height"),
        "ready_indices": state.get("ready_indices"),
        "infer_mode": state.get("infer_mode"),
        "infer_backend": state.get("infer_backend") or "local",
        "infer_device": state.get("infer_device"),
        "graph": graph,
        "looks": look_urls,
        "rgb_url": _file_url(run_id, "rgb.jpg"),
        "metric_url": _file_url(run_id, Path(state["metric_path"]).name)
        if state.get("metric_path")
        else None,
        "blend_url": _file_url(run_id, "blend_live.png"),
        "ply_url": _file_url(run_id, Path(state["ply_path"]).name)
        if state.get("ply_path")
        else None,
        "output_url": _file_url(run_id, "blend_live.png"),
        "preview_gif_url": _file_url(run_id, str(preview_gif))
        if preview_gif
        else None,
        "source_url": _file_url(run_id, str(source_file)) if source_file else None,
        "modal_usage": get_modal_usage(),
    }
    if state.get("modal_billing"):
        out["modal_billing"] = state["modal_billing"]
    return out


def _bake_run_frames(run_id: str, graph: dict[str, Any]) -> dict[str, Any]:
    """Evaluate graph for every ready frame into frames/NNNNNN/out.png."""
    state = get_run(run_id)
    if state is None:
        raise KeyError(run_id)
    out_dir = Path(state["out_dir"])
    meta = load_frames_meta(out_dir)
    count = int(state.get("frame_count") or 1)
    ready = int(
        (meta or {}).get("ready_count")
        or state.get("ready_count")
        or 0
    )
    if ready < 1:
        raise ValueError("No depth frames ready to bake")
    ready = min(ready, count)
    prev = int(state.get("frame_index") or 0)
    temporal_state: dict[str, Any] = {}
    frame_urls: list[str] = []
    try:
        for i in range(ready):
            set_frame(run_id, i)
            evaluate_graph(run_id, graph, temporal_state=temporal_state)
            blend = out_dir / "blend_live.png"
            if not blend.is_file():
                raise RuntimeError(f"Missing blend for frame {i}")
            dest = out_dir / "frames" / f"{i:06d}" / "out.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(blend, dest)
            frame_urls.append(_file_url(run_id, f"frames/{i:06d}/out.png"))
    finally:
        try:
            set_frame(run_id, min(prev, ready - 1))
            evaluate_graph(run_id, graph)
        except Exception:
            pass
    update_run(run_id, baked_count=ready, frame_urls=frame_urls)
    return {
        "run_id": run_id,
        "frame_count": count,
        "ready_count": ready,
        "baked_count": ready,
        "frame_urls": frame_urls,
        "output_url": frame_urls[0] if frame_urls else _file_url(run_id, "blend_live.png"),
    }


def _start_prepared_infer_job(
    run_id: str,
    *,
    model_key: Optional[str],
    max_size: Optional[int],
    infer_backend: Optional[str],
) -> str:
    job_id = uuid.uuid4().hex[:12]
    _set_job(
        job_id,
        status="running",
        progress=0.0,
        message="Queued…",
        kind="video",
        run_id=run_id,
        result=None,
        error=None,
    )

    def worker() -> None:
        published_run_id: Optional[str] = run_id
        try:
            state0 = get_run(run_id)
            if state0 is None:
                raise KeyError(run_id)
            out_dir = Path(state0["out_dir"])

            def on_progress(p: float, msg: str) -> None:
                _set_job(job_id, progress=float(p), message=msg, status="running")

            def on_partial(
                ready: int,
                total: int,
                partial_state: Optional[dict] = None,
                ready_indices: Optional[list] = None,
            ) -> None:
                nonlocal published_run_id
                prog = float(ready) / float(max(total, 1))
                idx = list(ready_indices) if ready_indices is not None else list(
                    range(ready)
                )
                mode = (
                    "modal_parallel"
                    if str(
                        infer_backend
                        or state0.get("infer_backend")
                        or (partial_state or {}).get("infer_backend")
                        or ""
                    ).lower()
                    == "modal"
                    else "local_sequential"
                )
                if partial_state is not None:
                    merged = dict(state0)
                    merged.update(partial_state)
                    merged["out_dir"] = str(out_dir.resolve())
                    put_run(merged)
                    result = _infer_response(merged)
                    published_run_id = result["run_id"]
                    update_run(
                        published_run_id,
                        ready_count=ready,
                        ready_indices=idx,
                        infer_status="running" if ready < total else "done",
                        infer_mode=mode,
                    )
                    result["ready_count"] = ready
                    result["ready_indices"] = idx
                    result["infer_mode"] = mode
                    result["infer_status"] = (
                        "running" if ready < total else "done"
                    )
                    _set_job(
                        job_id,
                        status="running",
                        phase="preview",
                        progress=max(0.05, prog * 0.95),
                        message=f"Preview ready · depth {ready}/{total}…",
                        run_id=published_run_id,
                        result=result,
                        ready_frames=ready,
                        total_frames=total,
                        ready_indices=idx,
                        infer_mode=mode,
                        error=None,
                    )
                elif published_run_id is not None:
                    update_run(
                        published_run_id,
                        ready_count=ready,
                        ready_indices=idx,
                        infer_status="running" if ready < total else "done",
                        _smooth_window_cache=None,
                        infer_mode=mode,
                    )
                    job = _get_job(job_id) or {}
                    result = dict(job.get("result") or {})
                    if result:
                        result["ready_count"] = ready
                        result["frame_count"] = total
                        result["ready_indices"] = idx
                        result["infer_mode"] = mode
                        result["infer_status"] = (
                            "running" if ready < total else "done"
                        )
                    _set_job(
                        job_id,
                        status="running",
                        phase="preview" if ready < total else "finishing",
                        progress=max(0.05, prog * 0.95),
                        message=f"Depth frame {ready}/{total}…",
                        ready_frames=ready,
                        total_frames=total,
                        ready_indices=idx,
                        infer_mode=mode,
                        result=result or job.get("result"),
                    )
                else:
                    _set_job(
                        job_id,
                        progress=max(0.05, prog * 0.95),
                        message=f"Depth frame {ready}/{total}…",
                        ready_frames=ready,
                        total_frames=total,
                        ready_indices=idx,
                        infer_mode=mode,
                    )

            state = infer_prepared_run(
                out_dir,
                model_key=model_key,
                max_size=max_size,
                infer_backend=infer_backend,
                on_progress=on_progress,
                on_partial=on_partial,
            )
            ready_final = int(
                state.get("ready_count") or state.get("frame_count") or 1
            )
            total_final = int(state.get("frame_count") or 1)
            put_run(state)
            update_run(
                run_id,
                ready_count=ready_final,
                infer_status="done",
                modal_billing=state.get("modal_billing"),
                depth=state.get("depth"),
                rgb=state.get("rgb"),
                looks=state.get("looks"),
                cliff=state.get("cliff"),
                look_paths=state.get("look_paths"),
                metric_path=state.get("metric_path"),
                blend_path=state.get("blend_path"),
            )
            result = _infer_response(get_run(run_id) or state)
            result["ready_count"] = ready_final
            result["frame_count"] = total_final
            result["infer_status"] = "done"
            if state.get("modal_billing"):
                result["modal_billing"] = state["modal_billing"]
            _set_job(
                job_id,
                status="done",
                phase="done",
                progress=1.0,
                message="Done",
                run_id=run_id,
                result=result,
                ready_frames=ready_final,
                total_frames=total_final,
                error=None,
                modal_billing=state.get("modal_billing"),
            )
        except Exception as e:
            _set_job(
                job_id,
                status="error",
                progress=1.0,
                message=str(e),
                error=f"{e}\n{traceback.format_exc()}",
            )

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def _set_job(job_id: str, **kwargs: Any) -> None:
    with _JOBS_LOCK:
        cur = dict(_JOBS.get(job_id) or {"job_id": job_id})
        cur.update(kwargs)
        _JOBS[job_id] = cur


def _get_job(job_id: str) -> Optional[dict[str, Any]]:
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        return dict(j) if j else None


@router.get("/meta")
def meta():
    from server.graph_eval import _look_meta_entry

    return {
        "models": [{"label": label, "key": key} for label, key in MODEL_CHOICES],
        "default_model": DEFAULT_MODEL,
        "max_sizes": [518, 756, 1008, 1536],
        "default_max_size": MAX_SIZE if MAX_SIZE in (518, 756, 1008, 1536) else 756,
        "infer_backends": ["local", "modal"],
        "default_infer_backend": DEFAULT_INFER_BACKEND,
        "modal_available": modal_ready(),
        "modal_pricing": pricing_meta(),
        "looks": [
            _look_meta_entry(k, v)
            for k, v in LOOK_PRESETS.items()
            if k != "scenery"
        ],
        "blend_modes": [{"key": k, "label": v} for k, v in BLEND_MODES.items()],
        "scenery_modes": [
            {"key": k, "label": v} for k, v in SCENERY_MATTE_MODES.items()
        ],
        "cmaps": ["gray", "magma", "inferno", "viridis", "turbo", "photo", "wash"],
        "node_types": [
            "depth",
            "video_in",
            "video_out",
            "smooth",
            "freq_fuse",
            "look",
            "scene",
            "blend",
            "relief",
            "output",
            "arena",
            "arena_out",
        ],
        "relief": {
            "default_gc_alpha": DEFAULT_RELIEF_GC_ALPHA,
            "default_invert": DEFAULT_RELIEF_INVERT,
            "default_max_side": DEFAULT_RELIEF_MAX_SIDE,
            "max_sides": list(RELIEF_MAX_SIDE_CHOICES),
        },
        "smooth": smooth_meta(),
        "freq_fuse": freq_fuse_meta(),
        "arena": auth_status(),
        "has_custom_default": load_default_template("still") is not None,
        "has_custom_video_default": load_default_template("video") is not None,
        "video_max_frames": 60,
        "video_target_fps": 10,
        "operator": operator_meta(),
        "has_custom_operator_default": get_operator_default()["has_custom"],
        "film": film_meta(),
    }


@router.get("/modal/usage")
def modal_usage():
    return get_modal_usage()


@router.post("/modal/usage/reset")
def modal_usage_reset():
    return reset_modal_usage()


@router.get("/health")
def health():
    return {"ok": True}


class UiHeartbeatBody(BaseModel):
    mode: Optional[str] = None
    last_evaluate_at: Optional[str] = None
    last_run_id: Optional[str] = None


@router.get("/status")
async def status(deep: bool = False):
    """Aggregated Modal / Local / UI / API / Blender MCP status."""
    return await build_status_async(deep=bool(deep))


@router.post("/status/ui")
def status_ui(body: UiHeartbeatBody):
    return {
        "ok": True,
        "ui": record_ui_heartbeat(
            mode=body.mode,
            last_evaluate_at=body.last_evaluate_at,
            last_run_id=body.last_run_id,
        ),
    }


@router.get("/cache")
def cache_info():
    from depth_core.store import cache_stats
    from server.cache import ram_stats

    return {**cache_stats(), **ram_stats()}


@router.post("/cache/prune")
def cache_prune():
    from depth_core.store import enforce_disk_budget
    from server.status import pinned_run_ids

    result = enforce_disk_budget(pinned_run_ids())
    from depth_core.store import cache_stats
    from server.cache import ram_stats

    return {"ok": True, **result, **cache_stats(refresh=True), **ram_stats()}


@router.get("/graph/default")
def get_default_graph(kind: str = "still"):
    kind = "video" if kind == "video" else "still"
    template = load_default_template(kind)
    if template is not None:
        graph = template_to_graph(template, "")
    else:
        graph = default_graph("", kind=kind)
    return {
        "has_custom": template is not None,
        "kind": kind,
        "graph": graph,
    }


@router.put("/graph/default")
def put_default_graph(body: GraphBody, kind: str = "still"):
    kind = "video" if kind == "video" else "still"
    path = save_default_template(body.model_dump(), kind=kind)
    return {
        "ok": True,
        "path": str(path),
        "kind": kind,
        "has_custom": True,
    }


@router.delete("/graph/default")
def delete_default_graph(kind: str = "still"):
    kind = "video" if kind == "video" else "still"
    cleared = clear_default_template(kind)
    return {"ok": True, "cleared": cleared, "kind": kind, "has_custom": False}


@router.get("/video/recipes")
def video_recipes():
    from server.video_recipes import list_video_recipes

    return {"recipes": list_video_recipes()}


@router.post("/infer/rgb")
async def infer_rgb(
    file: UploadFile = File(...),
    model_key: str = Form(DEFAULT_MODEL),
    max_size: int = Form(756),
    infer_backend: Optional[str] = Form(None),
):
    key = normalize_model_key(model_key)
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = Path(tmp.name)
    backend = _begin_infer(infer_backend)
    try:
        state = run_rgb(str(path), key, int(max_size), infer_backend=infer_backend)
    except ValueError as e:
        _end_infer(backend, error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        _end_infer(backend, error=str(e))
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        _end_infer(backend, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        path.unlink(missing_ok=True)
    return _infer_response(state)


@router.post("/arena/resolve")
def arena_resolve(body: ArenaResolveBody):
    """Parse an Are.na link and return block / channel metadata (+ preview pick)."""
    try:
        return resolve_ref(body.url, mode=body.mode or "auto")
    except ArenaError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/arena/me")
def arena_me():
    status = auth_status()
    if not status.get("authenticated"):
        raise HTTPException(
            status_code=401,
            detail=status.get("error")
            or "Are.na not authenticated — set ARENA_ACCESS_TOKEN in .env",
        )
    return status


@router.get("/arena/channels")
def arena_channels(per: int = 24):
    try:
        channels = list_recent_channels(per=per)
    except ArenaError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"channels": channels, **auth_status()}


@router.post("/arena/kitchens")
def arena_kitchens(body: ArenaKitchensBody):
    """Scan an Are.na channel or block for embedded Depth Ledger kitchen recipes."""
    try:
        return scan_arena_kitchens(
            body.url, limit=max(1, min(80, int(body.limit or 40)))
        )
    except ArenaError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/arena/publish")
def arena_publish(body: ArenaPublishBody):
    """Upload the run's output image into an Are.na channel."""
    state = get_run(body.run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    name = Path(body.filename or "blend_live.png").name
    path = Path(state["out_dir"]) / name
    if not path.is_file():
        # Fallback common aliases
        for alt in ("blend_live.png", "blend.png", "rgb.jpg"):
            cand = Path(state["out_dir"]) / alt
            if cand.is_file():
                path = cand
                break
    if not path.is_file():
        raise HTTPException(
            status_code=400,
            detail="No output image to publish — evaluate the graph first",
        )
    try:
        fields = build_arena_publish_fields(
            state,
            graph=state.get("graph"),
            user_title=body.title,
        )
        result = publish_image_to_channel(
            path,
            body.channel_id,
            title=fields["title"],
            description=fields["description"],
            original_source_url=fields.get("original_source_url"),
            original_source_title=fields.get("original_source_title"),
            metadata=fields.get("metadata"),
        )
    except ArenaError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "ok": True,
        "block": result,
        "file": path.name,
        "title": fields["title"],
        "run_label": fields.get("run_label"),
    }


@router.post("/infer/arena")
def infer_arena(body: ArenaInferBody):
    """Download an Are.na block image (or shuffled channel pick) and run depth."""
    key = normalize_model_key(body.model_key)
    path: Optional[Path] = None
    resolved: dict[str, Any] = {}
    backend = _begin_infer(body.infer_backend)
    state: dict[str, Any] = {}
    try:
        resolved = resolve_ref(body.url, mode=body.mode or "auto")
        block = resolved.get("block") or {}
        path = download_block_image(block)
        state = run_rgb(
            str(path),
            key,
            int(body.max_size),
            infer_backend=body.infer_backend,
            arena_block_id=block.get("id"),
        )
        state["arena"] = {
            "kind": resolved.get("kind"),
            "mode": resolved.get("mode"),
            "block": resolved.get("block"),
            "channel": resolved.get("channel"),
            "imageable_count": resolved.get("imageable_count"),
            "input": resolved.get("input"),
            "cached": resolved.get("cached", False),
        }
    except ArenaError as e:
        _end_infer(backend, error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        _end_infer(backend, error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        _end_infer(backend, error=str(e))
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        _end_infer(backend, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if path is not None:
            path.unlink(missing_ok=True)

    result = _infer_response(state)
    result["arena"] = state.get("arena") or {
        "kind": resolved.get("kind"),
        "mode": resolved.get("mode"),
        "block": resolved.get("block"),
        "channel": resolved.get("channel"),
        "imageable_count": resolved.get("imageable_count"),
        "input": resolved.get("input"),
        "cached": resolved.get("cached", False),
    }
    return result


@router.post("/batch/arena-roll")
def batch_arena_roll(body: ArenaRollBody):
    """Background job: shuffle N Are.na images → parallel/local infer → randomize → publish."""
    if not body.channel_id.strip():
        raise HTTPException(status_code=400, detail="channel_id required")
    if not body.arena_url.strip():
        raise HTTPException(status_code=400, detail="arena_url required")
    job_id = start_arena_roll_job(
        arena_url=body.arena_url.strip(),
        count=body.count,
        model_key=body.model_key,
        max_size=body.max_size,
        infer_backend=body.infer_backend,
        wackiness=body.wackiness,
        max_fidelity=body.max_fidelity,
        channel_id=body.channel_id.strip(),
        title=body.title or "",
        graph=body.graph.model_dump(),
        set_job=_set_job,
        get_job=_get_job,
    )
    return {"job_id": job_id}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    _set_job(job_id, cancel=True, message="Cancel requested…")
    return {"ok": True, "job_id": job_id}


@router.post("/infer/zip")
async def infer_zip(
    file: UploadFile = File(...),
    model_key: str = Form(DEFAULT_MODEL),
    max_size: int = Form(756),
    infer_backend: Optional[str] = Form(None),
):
    key = normalize_model_key(model_key)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(await file.read())
        path = Path(tmp.name)
    backend = _begin_infer(infer_backend)
    try:
        state = run_zip(str(path), key, int(max_size), infer_backend=infer_backend)
    except ValueError as e:
        _end_infer(backend, error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        _end_infer(backend, error=str(e))
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        _end_infer(backend, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        path.unlink(missing_ok=True)
    return _infer_response(state)


@router.post("/infer/pair")
async def infer_pair(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    model_key: str = Form("promptda-small"),
    max_size: int = Form(756),
    infer_backend: Optional[str] = Form(None),
):
    key = normalize_model_key(model_key)
    rgb_suffix = Path(rgb.filename or "rgb.jpg").suffix or ".jpg"
    depth_suffix = Path(depth.filename or "depth.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=rgb_suffix, delete=False) as tmp_rgb:
        tmp_rgb.write(await rgb.read())
        rgb_path = Path(tmp_rgb.name)
    with tempfile.NamedTemporaryFile(suffix=depth_suffix, delete=False) as tmp_d:
        tmp_d.write(await depth.read())
        depth_path = Path(tmp_d.name)
    backend = _begin_infer(infer_backend)
    try:
        state = run_pair(
            str(rgb_path),
            str(depth_path),
            key,
            int(max_size),
            infer_backend=infer_backend,
        )
    except ValueError as e:
        _end_infer(backend, error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        _end_infer(backend, error=str(e))
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        _end_infer(backend, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        rgb_path.unlink(missing_ok=True)
        depth_path.unlink(missing_ok=True)
    return _infer_response(state)


@router.post("/video/prepare")
async def video_prepare(
    file: UploadFile = File(...),
    model_key: str = Form("depth-pro"),
    max_size: int = Form(756),
    infer_backend: Optional[str] = Form(None),
):
    """Extract RGB frames + cheap preview GIF; no depth yet."""
    key = normalize_model_key(model_key)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = Path(tmp.name)
    try:
        state = prepare_video(
            path,
            key,
            int(max_size),
            infer_backend=infer_backend,
        )
        return _infer_response(state, evaluate=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        path.unlink(missing_ok=True)


class InferPreparedBody(BaseModel):
    model_key: Optional[str] = None
    max_size: Optional[int] = None
    infer_backend: Optional[str] = None


@router.post("/runs/{run_id}/infer")
def infer_prepared_video(run_id: str, body: InferPreparedBody = InferPreparedBody()):
    """Kick off depth infer for an already-prepared video run."""
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if int(state.get("frame_count") or 1) < 1:
        raise HTTPException(status_code=400, detail="Run has no frames")
    status = str(state.get("infer_status") or "")
    if status == "running":
        raise HTTPException(status_code=409, detail="Infer already running")
    job_id = _start_prepared_infer_job(
        run_id,
        model_key=body.model_key,
        max_size=body.max_size,
        infer_backend=body.infer_backend,
    )
    return {"job_id": job_id, "run_id": run_id}


@router.post("/runs/{run_id}/bake")
def bake_run_video(run_id: str, body: ExportBody):
    """Bake kitchen graph for every ready frame into frames/*/out.png."""
    try:
        return _bake_run_frames(run_id, body.graph.model_dump())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/infer/video")
async def infer_video(
    file: UploadFile = File(...),
    model_key: str = Form("depth-pro"),
    max_size: int = Form(756),
    infer_backend: Optional[str] = Form(None),
):
    """Start background Depth Pro video job; poll GET /api/jobs/{job_id}.

    Legacy one-shot path (prepare+infer). Prefer /api/video/prepare then
    /api/runs/{id}/infer for the Video In kitchen flow.
    """
    key = normalize_model_key(model_key)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = Path(tmp.name)

    job_id = uuid.uuid4().hex[:12]
    _set_job(
        job_id,
        status="running",
        progress=0.0,
        message="Queued…",
        kind="video",
        run_id=None,
        result=None,
        error=None,
    )

    def worker() -> None:
        published_run_id: Optional[str] = None

        try:
            def on_progress(p: float, msg: str) -> None:
                _set_job(job_id, progress=float(p), message=msg, status="running")

            def on_partial(
                ready: int,
                total: int,
                partial_state: Optional[dict] = None,
                ready_indices: Optional[list] = None,
            ) -> None:
                nonlocal published_run_id
                prog = float(ready) / float(max(total, 1))
                idx = list(ready_indices) if ready_indices is not None else list(
                    range(ready)
                )
                mode = (
                    "modal_parallel"
                    if (infer_backend or "").lower() == "modal"
                    else "local_sequential"
                )
                if partial_state is not None and published_run_id is None:
                    # First frame ready — open kitchen while remaining frames infer.
                    result = _infer_response(partial_state)
                    published_run_id = result["run_id"]
                    update_run(
                        published_run_id,
                        ready_count=ready,
                        ready_indices=idx,
                        infer_status="running" if ready < total else "done",
                        infer_mode=mode,
                    )
                    result["ready_count"] = ready
                    result["ready_indices"] = idx
                    result["infer_mode"] = mode
                    result["infer_status"] = (
                        "running" if ready < total else "done"
                    )
                    _set_job(
                        job_id,
                        status="running",
                        phase="preview",
                        progress=max(0.05, prog * 0.95),
                        message=f"Preview ready · depth {ready}/{total}…",
                        run_id=published_run_id,
                        result=result,
                        ready_frames=ready,
                        total_frames=total,
                        ready_indices=idx,
                        infer_mode=mode,
                        error=None,
                    )
                elif published_run_id is not None:
                    update_run(
                        published_run_id,
                        ready_count=ready,
                        ready_indices=idx,
                        infer_status="running" if ready < total else "done",
                        _smooth_window_cache=None,
                        infer_mode=mode,
                    )
                    # Refresh result ready_count for pollers without rebuilding graph.
                    job = _get_job(job_id) or {}
                    result = dict(job.get("result") or {})
                    if result:
                        result["ready_count"] = ready
                        result["frame_count"] = total
                        result["ready_indices"] = idx
                        result["infer_mode"] = mode
                        result["infer_status"] = (
                            "running" if ready < total else "done"
                        )
                    _set_job(
                        job_id,
                        status="running",
                        phase="preview" if ready < total else "finishing",
                        progress=max(0.05, prog * 0.95),
                        message=f"Depth frame {ready}/{total}…",
                        ready_frames=ready,
                        total_frames=total,
                        ready_indices=idx,
                        infer_mode=mode,
                        result=result or job.get("result"),
                    )
                else:
                    _set_job(
                        job_id,
                        progress=max(0.05, prog * 0.95),
                        message=f"Depth frame {ready}/{total}…",
                        ready_frames=ready,
                        total_frames=total,
                        ready_indices=idx,
                        infer_mode=mode,
                    )

            state = run_video(
                path,
                key,
                int(max_size),
                on_progress=on_progress,
                on_partial=on_partial,
                infer_backend=infer_backend,
            )
            ready_final = int(
                state.get("ready_count") or state.get("frame_count") or 1
            )
            total_final = int(state.get("frame_count") or 1)
            if published_run_id is not None:
                # Keep the live kitchen graph — only mark infer complete.
                update_run(
                    published_run_id,
                    ready_count=ready_final,
                    infer_status="done",
                    modal_billing=state.get("modal_billing"),
                )
                job = _get_job(job_id) or {}
                result = dict(job.get("result") or {})
                result["run_id"] = published_run_id
                result["ready_count"] = ready_final
                result["frame_count"] = total_final
                result["infer_status"] = "done"
                if state.get("modal_billing"):
                    result["modal_billing"] = state["modal_billing"]
                _set_job(
                    job_id,
                    status="done",
                    phase="done",
                    progress=1.0,
                    message="Done",
                    run_id=published_run_id,
                    result=result,
                    ready_frames=ready_final,
                    total_frames=total_final,
                    error=None,
                    modal_billing=state.get("modal_billing"),
                )
            else:
                result = _infer_response(state)
                update_run(
                    result["run_id"],
                    ready_count=ready_final,
                    infer_status="done",
                )
                result["ready_count"] = ready_final
                result["infer_status"] = "done"
                _set_job(
                    job_id,
                    status="done",
                    phase="done",
                    progress=1.0,
                    message="Done",
                    run_id=result["run_id"],
                    result=result,
                    ready_frames=ready_final,
                    total_frames=total_final,
                    error=None,
                    modal_billing=result.get("modal_billing"),
                )
        except ValueError as e:
            _set_job(
                job_id,
                status="error",
                progress=1.0,
                message=str(e),
                error=str(e),
            )
        except RuntimeError as e:
            _set_job(
                job_id,
                status="error",
                progress=1.0,
                message=str(e),
                error=str(e),
            )
        except Exception as e:
            _set_job(
                job_id,
                status="error",
                progress=1.0,
                message=str(e),
                error=f"{e}\n{traceback.format_exc()}",
            )
        finally:
            path.unlink(missing_ok=True)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "message": job.get("message"),
        "run_id": job.get("run_id"),
        "error": job.get("error"),
        "result": job.get("result"),
        "kind": job.get("kind"),
        "phase": job.get("phase"),
        "i": job.get("i"),
        "n": job.get("n"),
        "saved": job.get("saved"),
        "ready_frames": job.get("ready_frames"),
        "total_frames": job.get("total_frames"),
        "ready_indices": job.get("ready_indices"),
        "infer_mode": job.get("infer_mode"),
        "modal_billing": job.get("modal_billing"),
        "modal_usage": get_modal_usage(),
    }


@router.post("/runs/{run_id}/evaluate")
def evaluate(run_id: str, body: EvaluateBody):
    try:
        result = evaluate_graph(run_id, body.graph.model_dump())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return result


@router.get("/operator/default")
def get_default_operator():
    return get_operator_default()


@router.put("/operator/default")
def put_default_operator(body: OperatorPatchBody):
    path = save_default_operator_patch(body.to_patch())
    return {"ok": True, "path": str(path), "has_custom": True}


@router.delete("/operator/default")
def delete_default_operator():
    cleared = clear_default_operator_patch()
    return {"ok": True, "cleared": cleared, "has_custom": False}


@router.post("/runs/{run_id}/operator/evaluate")
def operator_evaluate(run_id: str, body: OperatorPatchBody):
    try:
        return evaluate_operator(run_id, body.to_patch())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/runs/{run_id}/operator/export")
def operator_export(run_id: str, body: OperatorPatchBody):
    try:
        return export_operator_video(run_id, body.to_patch())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/runs/{run_id}/operator/stream")
def operator_stream(run_id: str):
    """Latest operator_live as JPEG for poll / MJPEG consumers."""
    from fastapi.responses import Response

    data = operator_stream_jpeg(run_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No operator plate — evaluate Operator first",
        )
    # File is PNG; convert lightly or serve as png
    path_state = get_run(run_id)
    if path_state:
        png = Path(path_state["out_dir"]) / "operator_live.png"
        if png.is_file():
            return FileResponse(png, media_type="image/png")
    return Response(content=data, media_type="image/png")


@router.post("/runs/{run_id}/frame")
def set_run_frame(run_id: str, body: FrameBody):
    try:
        set_frame(run_id, int(body.index))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    state = get_run(run_id)
    assert state is not None
    # Keep root rgb alias in sync for clients that fetch rgb.jpg
    out_dir = Path(state["out_dir"])
    src = out_dir / "frames" / f"{int(body.index):06d}" / "rgb.jpg"
    if src.is_file():
        shutil.copyfile(src, out_dir / f"{run_id}_rgb.jpg")

    result: dict[str, Any] = {
        "run_id": run_id,
        "frame_index": int(state.get("frame_index") or 0),
        "frame_count": int(state.get("frame_count") or 1),
        "rgb_url": _file_url(run_id, "rgb.jpg"),
        "output_url": _file_url(run_id, "blend_live.png"),
        "node_previews": {},
    }
    graph = body.graph.model_dump() if body.graph is not None else state.get("graph")
    if graph is None:
        recipe = load_recipe(out_dir)
        kind = "video" if int(state.get("frame_count") or 1) > 1 else "still"
        graph = (recipe or {}).get("graph") or default_graph(run_id, kind=kind)
    try:
        ev = evaluate_graph(run_id, graph)
        result["output_url"] = ev["output_url"]
        result["node_previews"] = ev.get("node_previews") or {}
        result["elapsed_ms"] = ev.get("elapsed_ms")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return result


class BlenderExportBody(BaseModel):
    preset: str = "ledger"
    write_inv: bool = True
    write_matte: bool = True
    write_relief: bool = False
    scenery_mode: str = "smart"
    scenery_protect: float = 0.22
    scenery_feather: float = 0.35
    relief_gc_alpha: float = 10.0
    relief_invert: bool = False
    relief_max_side: int = 512


@router.post("/runs/{run_id}/export/blender")
def export_run_blender(run_id: str, body: BlenderExportBody = BlenderExportBody()):
    """Write float meters EXR + near_far.json (+ optional inv/matte/relief) for Blender."""
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    depth = state.get("depth")
    if depth is None:
        raise HTTPException(status_code=400, detail="Run has no depth yet")

    from depth_core.blender_export import (
        export_blender_bundle,
        export_matte_exr,
        export_relief_png,
    )
    from depth_core.pipeline import (
        DEFAULT_SCENERY_FEATHER,
        DEFAULT_SCENERY_MATTE,
        DEFAULT_SCENERY_PROTECT,
        _scenery_depth_window,
        _subject_hi_pct,
        _subject_matte,
    )

    out_dir = Path(state["out_dir"])
    try:
        bundle = export_blender_bundle(
            out_dir,
            run_id,
            depth,
            rgb=state.get("rgb"),
            cliff=state.get("cliff"),
            preset=body.preset or "ledger",
            write_inv=bool(body.write_inv),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    matte_url = None
    if body.write_matte:
        try:
            cliff = float(state.get("cliff") or _subject_hi_pct(depth))
            near, _far, cut = _scenery_depth_window(depth, cliff)
            matte = _subject_matte(
                depth,
                cut,
                near,
                protect=float(body.scenery_protect or DEFAULT_SCENERY_PROTECT),
                feather=float(body.scenery_feather or DEFAULT_SCENERY_FEATHER),
                mode=body.scenery_mode or DEFAULT_SCENERY_MATTE,
            )
            # Keep-matte for scenery (1 = keep far / 0 = punch subject)
            keep = 1.0 - np.clip(matte, 0.0, 1.0)
            mpath = export_matte_exr(out_dir, run_id, keep)
            matte_url = _file_url(run_id, mpath.name)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Matte export failed: {e}"
            ) from e

    relief_url = None
    if body.write_relief:
        try:
            plate = state.get("looks", {}).get("ledger") if state.get("looks") else None
            if plate is None:
                from depth_core.pipeline import (
                    DEFAULT_LOOK_A,
                    LOOK_PRESETS,
                    _prepare_inv_smooth,
                    _resolve_hi_pct,
                    _resolve_lo_pct,
                    _subject_hi_pct,
                    _visualize_inverse_depth,
                )

                cfg = LOOK_PRESETS.get("ledger") or LOOK_PRESETS[DEFAULT_LOOK_A]
                cliff = float(state.get("cliff") or _subject_hi_pct(depth))
                _, inv_smooth = _prepare_inv_smooth(depth)
                plate = _visualize_inverse_depth(
                    depth,
                    cmap=cfg["cmap"],
                    lo_pct=_resolve_lo_pct(cfg, cliff),
                    hi_pct=_resolve_hi_pct(cfg, cliff),
                    gamma=cfg["gamma"],
                    grade=cfg.get("grade"),
                    inv_smooth=inv_smooth,
                    photo=state.get("rgb") if cfg["cmap"] in ("photo", "wash") else None,
                )
            if plate is None:
                raise ValueError("No ledger look plate for relief")
            rpath = export_relief_png(
                out_dir,
                run_id,
                plate,
                gc_alpha=float(body.relief_gc_alpha),
                invert=bool(body.relief_invert),
                max_side=int(body.relief_max_side),
            )
            relief_url = _file_url(run_id, rpath.name)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Relief export failed: {e}"
            ) from e

    return {
        "ok": True,
        "run_id": run_id,
        "depth_exr_url": _file_url(run_id, Path(bundle["depth_exr"]).name),
        "inv_smooth_exr_url": (
            _file_url(run_id, Path(bundle["inv_smooth_exr"]).name)
            if bundle.get("inv_smooth_exr")
            else None
        ),
        "near_far_url": _file_url(run_id, Path(bundle["near_far_json"]).name),
        "rgb_url": _file_url(run_id, "rgb.jpg") if bundle.get("rgb") else None,
        "matte_exr_url": matte_url,
        "relief_url": relief_url,
        "near_far": bundle.get("near_far"),
    }


@router.get("/runs/{run_id}/export/npz")
def export_run_npz(run_id: str):
    """Float depth (m) + inv_smooth + valid mask + rgb/output plates as one NPZ."""
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    depth = state.get("depth")
    if depth is None:
        raise HTTPException(status_code=400, detail="Run has no depth yet")

    from depth_core.blender_export import export_npz

    out_dir = Path(state["out_dir"])
    blend_path = Path(state.get("blend_path") or out_dir / "blend_live.png")
    output = imageio.imread(blend_path) if blend_path.is_file() else None
    try:
        path = export_npz(
            out_dir,
            run_id,
            depth,
            rgb=state.get("rgb"),
            output=output,
            cliff=state.get("cliff"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NPZ export failed: {e}") from e
    return FileResponse(
        path, media_type="application/octet-stream", filename=path.name
    )


@router.get("/runs/{run_id}/export/strata")
def export_run_strata(
    run_id: str,
    mode: str = "relief",
    matte: bool = True,
    gc_alpha: float = 10.0,
    gamma: float = 1.0,
    max_side: int = 512,
):
    """8-bit grey PNG for Strata image slots: near subject = dark, background = white."""
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    depth = state.get("depth")
    if depth is None:
        raise HTTPException(status_code=400, detail="Run has no depth yet")

    from depth_core.strata_export import STRATA_MODES, export_strata_png

    if mode not in STRATA_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {STRATA_MODES}")
    try:
        path = export_strata_png(
            Path(state["out_dir"]),
            run_id,
            depth,
            cliff=state.get("cliff"),
            mode=mode,
            matte=bool(matte),
            gc_alpha=float(gc_alpha),
            gamma=float(gamma),
            max_side=int(np.clip(max_side, 64, 2048)),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strata export failed: {e}") from e
    return FileResponse(path, media_type="image/png", filename=path.name)


class BlenderOpenBody(BlenderExportBody):
    open: bool = True


@router.post("/runs/{run_id}/open/blender")
def open_run_blender(run_id: str, body: BlenderOpenBody = BlenderOpenBody()):
    """Export EXR/JSON then launch Blender 5.2 with DepthLedger_v1 + load_run."""
    export = export_run_blender(run_id, body)
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    from server.blender_launch import launch_blender_for_run

    try:
        launch = launch_blender_for_run(
            state["out_dir"],
            preset=body.preset or "ledger",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to launch Blender: {e}"
        ) from e
    return {**export, "opened": True, "launch": launch}


@router.post("/runs/{run_id}/export")
def export_run_video(run_id: str, body: ExportBody):
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")

    def _has_ffmpeg() -> bool:
        if shutil.which("ffmpeg"):
            return True
        try:
            import imageio_ffmpeg

            return bool(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            return False

    if not _has_ffmpeg():
        raise HTTPException(
            status_code=400,
            detail="ffmpeg not found. Install system ffmpeg or `pip install imageio-ffmpeg`.",
        )
    count = int(state.get("frame_count") or 1)
    if count < 1:
        raise HTTPException(status_code=400, detail="No frames to export")

    meta = load_frames_meta(Path(state["out_dir"]))
    ready = int(
        (meta or {}).get("ready_count")
        or state.get("ready_count")
        or count
    )
    if ready < count or str((meta or {}).get("infer_status") or "") == "running":
        raise HTTPException(
            status_code=409,
            detail=f"Video still inferring ({ready}/{count} frames ready). Wait for Done.",
        )

    out_dir = Path(state["out_dir"])
    export_dir = out_dir / "_export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = body.graph.model_dump()
    fps = float(state.get("sample_fps") or 10)

    # Prefer already-baked out.png frames when complete.
    baked = [
        out_dir / "frames" / f"{i:06d}" / "out.png" for i in range(count)
    ]
    if all(p.is_file() for p in baked):
        try:
            for i, src in enumerate(baked):
                shutil.copyfile(src, export_dir / f"{i:06d}.png")
            mp4 = export_frames_to_mp4(
                [export_dir / f"{i:06d}.png" for i in range(count)],
                export_dir / "out.mp4",
                fps,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {
            "run_id": run_id,
            "export_url": _file_url(run_id, "_export/out.mp4"),
            "frame_count": count,
            "fps": fps,
        }

    prev = int(state.get("frame_index") or 0)
    temporal_state: dict[str, Any] = {}
    try:
        for i in range(count):
            set_frame(run_id, i)
            evaluate_graph(run_id, graph, temporal_state=temporal_state)
            blend = out_dir / "blend_live.png"
            if not blend.is_file():
                raise RuntimeError(f"Missing blend for frame {i}")
            shutil.copyfile(blend, export_dir / f"{i:06d}.png")
            # Keep bake cache warm for Output filmstrip.
            shutil.copyfile(blend, out_dir / "frames" / f"{i:06d}" / "out.png")
        mp4 = export_frames_to_mp4(
            [export_dir / f"{i:06d}.png" for i in range(count)],
            export_dir / "out.mp4",
            fps,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        try:
            set_frame(run_id, prev)
            evaluate_graph(run_id, graph)
        except Exception:
            pass

    return {
        "run_id": run_id,
        "export_url": _file_url(run_id, "_export/out.mp4"),
        "frame_count": count,
        "fps": fps,
    }


@router.get("/runs")
def runs(limit: int = 40):
    items = []
    for label, path in list_recent_runs(limit):
        run_id = Path(path).name
        recipe = load_recipe(path)
        items.append(
            {
                "run_id": run_id,
                "label": label,
                "path": path,
                "has_recipe": recipe is not None,
                "recipe_version": (recipe or {}).get("version", 1),
            }
        )
    return {"runs": items}


@router.get("/runs/{run_id}")
def run_detail(run_id: str):
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    recipe = load_recipe(state["out_dir"])
    kind = state.get("kind") or (
        "video" if int(state.get("frame_count") or 1) > 1 else "still"
    )
    graph = (recipe or {}).get("graph") or state.get("graph") or default_graph(
        run_id, kind=kind
    )
    return {
        "run_id": run_id,
        "kind": state.get("kind") or "still",
        "frame_count": int(state.get("frame_count") or 1),
        "frame_index": int(state.get("frame_index") or 0),
        "sample_fps": float(state.get("sample_fps") or 10),
        "model_key": state.get("model_key") or (recipe or {}).get("model_key"),
        "graph": graph,
        "looks": {
            k: _file_url(run_id, Path(p).name)
            for k, p in (state.get("look_paths") or {}).items()
            if Path(p).exists()
        },
        "rgb_url": _file_url(run_id, "rgb.jpg"),
        "blend_url": _file_url(run_id, "blend_live.png"),
        "metric_url": _file_url(run_id, Path(state["metric_path"]).name)
        if state.get("metric_path")
        else None,
        "output_url": _file_url(run_id, "blend_live.png"),
    }


@router.get("/runs/{run_id}/depth-sample")
def depth_sample(
    run_id: str,
    u: float = 0.5,
    v: float = 0.5,
    frame: Optional[int] = None,
):
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    depth = state.get("depth")
    if depth is None:
        raise HTTPException(status_code=404, detail="No depth for run")

    frame_index = int(state.get("frame_index") or 0)
    if frame is not None and int(frame) != frame_index:
        try:
            state = set_frame(run_id, int(frame))
            depth = state.get("depth")
            frame_index = int(state.get("frame_index") or 0)
        except (KeyError, ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    if depth is None:
        raise HTTPException(status_code=404, detail="No depth for run")

    try:
        sample = depth_percentile_at(depth, u, v)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {**sample, "frame_index": frame_index}


@router.get("/runs/{run_id}/tune-preview")
def tune_preview(
    run_id: str,
    frame: Optional[int] = None,
    node_id: Optional[str] = None,
):
    """Full-size linear grayscale depth plate for Tune-windows picking."""
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if state.get("depth") is None:
        raise HTTPException(status_code=404, detail="No depth for run")

    frame_index = int(state.get("frame_index") or 0)
    if frame is not None and int(frame) != frame_index:
        try:
            set_frame(run_id, int(frame))
        except (KeyError, ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        return render_tune_turbo_preview(run_id, node_id=node_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/runs/{run_id}/recipe")
def get_recipe(run_id: str):
    out_dir = OUTPUT_DIR / run_id
    if not out_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    recipe = load_recipe(out_dir)
    if recipe is None:
        raise HTTPException(status_code=404, detail="No recipe")
    return recipe


@router.put("/runs/{run_id}/recipe")
def put_recipe(run_id: str, body: dict[str, Any]):
    out_dir = OUTPUT_DIR / run_id
    if not out_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    path = save_recipe(out_dir, body)
    return {"ok": True, "path": str(path)}


@router.get("/runs/{run_id}/files/{name:path}")
def get_file(run_id: str, name: str):
    out_dir = (OUTPUT_DIR / run_id).resolve()
    if not out_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    # Prevent path traversal
    target = (out_dir / name).resolve()
    if not str(target).startswith(str(out_dir)):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.is_file():
        aliases = {
            "blend.png": f"{run_id}_blend.png",
            "blend_live.png": f"{run_id}_blend.png",
            "rgb.jpg": f"{run_id}_rgb.jpg",
            "depth.npz": f"{run_id}_depth.npz",
            "depth_mm.png": f"{run_id}_depth_mm.png",
        }
        alt = aliases.get(name)
        if alt:
            target = out_dir / alt
        if not target.is_file() and name in ("rgb.jpg", f"{run_id}_rgb.jpg"):
            state = get_run(run_id)
            fi = int((state or {}).get("frame_index") or 0)
            frame_rgb = out_dir / "frames" / f"{fi:06d}" / "rgb.jpg"
            if frame_rgb.is_file():
                target = frame_rgb
            else:
                jpgs = sorted(out_dir.glob("*_rgb.jpg"))
                if jpgs:
                    target = jpgs[0]
        if not target.is_file():
            if name.startswith("look_") and name.endswith(".png"):
                key = name[len("look_") : -4]
                cand = out_dir / f"{run_id}_look_{key}.png"
                if cand.is_file():
                    target = cand
        if not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
    media = "application/octet-stream"
    suf = target.suffix.lower()
    if suf in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        media = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }[suf]
    elif suf == ".mp4":
        media = "video/mp4"
    elif suf in {".webm"}:
        media = "video/webm"
    elif suf in {".mov", ".m4v"}:
        media = "video/quicktime"
    return FileResponse(target, media_type=media)


@router.post("/warmup")
def warmup(model_key: str = DEFAULT_MODEL, infer_backend: Optional[str] = None):
    backend = resolve_backend(infer_backend)
    if backend == "modal":
        set_modal_state("ready")
        try:
            from depth_core.infer_modal import modal_warmup

            result = modal_warmup()
            set_modal_state("ready", last_job="warmup")
            set_warmup("ready")
            return {
                "ok": True,
                "infer_backend": "modal",
                "model_key": model_key,
                **result,
                "modal_usage": get_modal_usage(),
            }
        except RuntimeError as e:
            set_modal_state("error", last_error=str(e))
            set_warmup("error", error=str(e))
            raise HTTPException(status_code=503, detail=str(e)) from e
    set_warmup("warming")
    set_local_infer("loading")
    try:
        loaded = get_backend(model_key)
        set_warmup("ready")
        set_local_infer("idle", device=loaded.device)
        return {
            "ok": True,
            "infer_backend": "local",
            "model_key": loaded.key,
            "device": loaded.device,
            "kind": loaded.kind,
        }
    except Exception as e:
        set_warmup("error", error=str(e))
        set_local_infer("failed", last_error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# —— Film (flicker-free video) — dedicated stack, not kitchen / operator ——


class FilmParamsBody(BaseModel):
    model_key: Optional[str] = None
    max_size: Optional[int] = None
    infer_backend: Optional[str] = None
    anchor_focal: Optional[bool] = None
    focallength_px: Optional[float] = None
    focallength_override: Optional[float] = None
    vis_min_m: Optional[float] = None
    vis_max_m: Optional[float] = None
    bounds_locked: Optional[bool] = None
    ema_enabled: Optional[bool] = None
    ema_alpha: Optional[float] = None
    gamma: Optional[float] = None
    cmap: Optional[str] = None
    preview_frames: Optional[int] = None

    def stack_params(self) -> dict[str, Any]:
        data = self.model_dump(exclude_unset=True)
        for k in ("model_key", "max_size", "infer_backend"):
            data.pop(k, None)
        return data


def _film_dir(run_id: str) -> Path:
    out_dir = OUTPUT_DIR / run_id
    if not out_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    return out_dir


def _film_payload(out_dir: Path) -> dict[str, Any]:
    info = clip_info(out_dir)
    state = {
        "out_dir": str(out_dir.resolve()),
        "kind": "video",
        "pipeline": "film",
        "frame_count": info["frame_count"],
        "frame_index": 0,
        "ready_count": info["ready_count"],
        "infer_status": info["infer_status"],
        "sample_fps": info["sample_fps"],
        "model_key": info.get("model_key"),
        "max_size": info.get("max_size"),
        "frame_width": info.get("frame_width"),
        "frame_height": info.get("frame_height"),
        "source_file": info.get("source_file"),
        "preview_gif": info.get("preview_gif"),
        "film": info["params"],
    }
    run_id = put_run(state)
    vis_ready = list(info.get("vis_ready") or [])
    n = int(info.get("frame_count") or 0)
    live = out_dir / FILM_LIVE
    return {
        "run_id": run_id,
        **info,
        "rgb_url": _file_url(run_id, "rgb.jpg"),
        "output_url": _file_url(run_id, FILM_LIVE) if live.is_file() else None,
        "preview_gif_url": (
            _file_url(run_id, str(info["preview_gif"]))
            if info.get("preview_gif")
            else None
        ),
        "source_url": (
            _file_url(run_id, str(info["source_file"]))
            if info.get("source_file")
            else None
        ),
        "vis_urls": [_file_url(run_id, f"frames/{i:06d}/vis.png") for i in vis_ready],
        "rgb_urls": [_file_url(run_id, f"frames/{i:06d}/rgb.jpg") for i in range(n)],
        "modal_usage": get_modal_usage(),
        "modal_billing": (info.get("params") or {}).get("modal_billing"),
    }


def _start_film_job(
    run_id: str,
    kind: str,
    body: FilmParamsBody,
) -> str:
    out_dir = _film_dir(run_id)
    job_id = uuid.uuid4().hex[:12]
    _set_job(
        job_id,
        status="running",
        progress=0.0,
        message="Queued…",
        kind="film",
        phase=kind,
        run_id=run_id,
        result=None,
        error=None,
    )
    params = {**load_film_params(out_dir), **body.stack_params()}
    backend = _begin_infer(body.infer_backend)

    def worker() -> None:
        try:

            def on_progress(p: float, msg: str) -> None:
                job = _get_job(job_id) or {}
                if job.get("cancel"):
                    raise RuntimeError("cancelled")
                _set_job(
                    job_id,
                    progress=float(p),
                    message=msg,
                    status="running",
                    run_id=run_id,
                )

            def on_partial(
                ready: int,
                total: int,
                ready_indices: Optional[list[int]] = None,
            ) -> None:
                payload = _film_payload(out_dir)
                _set_job(
                    job_id,
                    progress=max(0.05, float(ready) / float(max(total, 1))),
                    message=f"Depth {ready}/{total}…",
                    result=payload,
                    ready_frames=ready,
                    total_frames=total,
                    ready_indices=list(ready_indices or []),
                    run_id=run_id,
                )

            fn = run_film_preview if kind == "preview" else run_film_full
            result_params = fn(
                out_dir,
                params,
                model_key=body.model_key,
                max_size=body.max_size,
                infer_backend=body.infer_backend,
                on_progress=on_progress,
                on_partial=on_partial,
            )
            payload = _film_payload(out_dir)
            if result_params.get("modal_billing"):
                payload["modal_billing"] = result_params["modal_billing"]
            _set_job(
                job_id,
                status="done",
                progress=1.0,
                message="Done",
                result=payload,
                run_id=run_id,
                error=None,
            )
            _end_infer(backend, state={"out_dir": str(out_dir)})
        except Exception as exc:
            err = str(exc)
            if err == "cancelled":
                _set_job(
                    job_id,
                    status="cancelled",
                    error="cancelled",
                    message="Cancelled",
                    run_id=run_id,
                )
                _end_infer(backend, state={"out_dir": str(out_dir)})
                return
            traceback.print_exc()
            _set_job(
                job_id,
                status="error",
                error=err,
                message=err,
                run_id=run_id,
            )
            _end_infer(backend, error=err)

    threading.Thread(target=worker, daemon=True).start()
    return job_id


@router.post("/film/prepare")
async def film_prepare(
    file: UploadFile = File(...),
    model_key: str = Form("depth-pro"),
    max_size: int = Form(756),
    infer_backend: Optional[str] = Form(None),
    target_fps: float = Form(10.0),
    max_frames: int = Form(60),
):
    """Extract RGB frames for the film stack. No kitchen graph is attached."""
    key = normalize_model_key(model_key)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = Path(tmp.name)
    try:
        state = prepare_film(
            path,
            key,
            int(max_size),
            target_fps=float(target_fps),
            max_frames=int(max_frames),
            infer_backend=infer_backend,
        )
        return _film_payload(Path(state["out_dir"]))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        path.unlink(missing_ok=True)


@router.get("/film/{run_id}")
def film_get(run_id: str):
    return _film_payload(_film_dir(run_id))


@router.post("/film/{run_id}/preview")
def film_preview(run_id: str, body: FilmParamsBody = FilmParamsBody()):
    job_id = _start_film_job(run_id, "preview", body)
    return {"job_id": job_id, "run_id": run_id}


@router.post("/film/{run_id}/infer")
def film_infer(run_id: str, body: FilmParamsBody = FilmParamsBody()):
    job_id = _start_film_job(run_id, "infer", body)
    return {"job_id": job_id, "run_id": run_id}


@router.post("/film/{run_id}/render")
def film_render(run_id: str, body: FilmParamsBody = FilmParamsBody()):
    out_dir = _film_dir(run_id)
    params = normalize_film_params({**load_film_params(out_dir), **body.stack_params()})
    if body.vis_min_m is not None or body.vis_max_m is not None:
        params["bounds_locked"] = True
    save_film_params(out_dir, params)
    try:
        render_film_vis(out_dir, params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return _film_payload(out_dir)


@router.post("/film/{run_id}/export")
def film_export(run_id: str):
    out_dir = _film_dir(run_id)
    info = clip_info(out_dir)
    n = int(info.get("frame_count") or 0)
    vis_n = len(info.get("vis_ready") or [])
    if vis_n < n or n < 1:
        raise HTTPException(
            status_code=409,
            detail=f"Render all frames before export ({vis_n}/{n} vis ready).",
        )

    def _has_ffmpeg() -> bool:
        if shutil.which("ffmpeg"):
            return True
        try:
            import imageio_ffmpeg

            return bool(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            return False

    if not _has_ffmpeg():
        raise HTTPException(
            status_code=400,
            detail="ffmpeg not found. Install system ffmpeg or `pip install imageio-ffmpeg`.",
        )
    try:
        mp4 = export_film_mp4(out_dir, fps=float(info.get("sample_fps") or 10))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        "run_id": run_id,
        "export_url": _file_url(run_id, "_export/film.mp4"),
        "frame_count": n,
        "fps": float(info.get("sample_fps") or 10),
        "path": str(mp4),
    }


@router.get("/film/{run_id}/sample")
def film_sample(
    run_id: str,
    u: float = 0.5,
    v: float = 0.5,
    frame: int = 0,
):
    out_dir = _film_dir(run_id)
    try:
        return sample_film_depth(out_dir, int(frame), float(u), float(v))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class BatchParamsBody(BaseModel):
    model_key: Optional[str] = None
    max_size: Optional[int] = None
    infer_backend: Optional[str] = None
    folder_name: Optional[str] = None
    anchor_focal: Optional[bool] = None
    focallength_px: Optional[float] = None
    focallength_override: Optional[float] = None
    vis_min_m: Optional[float] = None
    vis_max_m: Optional[float] = None
    bounds_locked: Optional[bool] = None
    gamma: Optional[float] = None
    cmap: Optional[str] = None

    def stack_params(self) -> dict[str, Any]:
        data = self.model_dump(exclude_unset=True)
        for k in ("model_key", "max_size", "infer_backend", "folder_name"):
            data.pop(k, None)
        return data


def _batch_dir(run_id: str) -> Path:
    out_dir = OUTPUT_DIR / run_id
    if not out_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    meta = load_frames_meta(out_dir) or {}
    if meta.get("kind") != "batch":
        raise HTTPException(status_code=404, detail="Batch not found")
    return out_dir


def _batch_payload(out_dir: Path) -> dict[str, Any]:
    info = batch_info(out_dir)
    state = {
        "out_dir": str(out_dir.resolve()),
        "kind": "batch",
        "pipeline": "batch",
        "frame_count": info["frame_count"],
        "frame_index": 0,
        "ready_count": info["ready_count"],
        "infer_status": info["infer_status"],
        "model_key": info.get("model_key"),
        "max_size": info.get("max_size"),
    }
    run_id = put_run(state)
    n = int(info.get("frame_count") or 0)
    return {
        "run_id": run_id,
        **info,
        "rgb_urls": [_file_url(run_id, f"frames/{i:06d}/rgb.jpg") for i in range(n)],
        "vis_urls": [
            _file_url(run_id, f"frames/{i:06d}/vis.png")
            for i in (info.get("vis_ready") or [])
        ],
        "modal_usage": get_modal_usage(),
        "modal_billing": (info.get("params") or {}).get("modal_billing"),
    }


def _start_batch_job(run_id: str, kind: str, body: BatchParamsBody) -> str:
    out_dir = _batch_dir(run_id)
    job_id = uuid.uuid4().hex[:12]
    _set_job(
        job_id,
        status="running",
        progress=0.0,
        message="Queued…",
        kind="batch",
        phase=kind,
        run_id=run_id,
        result=None,
        error=None,
    )
    params = {**load_film_params(out_dir), **body.stack_params()}
    backend = _begin_infer(body.infer_backend)

    def worker() -> None:
        try:

            def on_progress(p: float, msg: str) -> None:
                job = _get_job(job_id) or {}
                if job.get("cancel"):
                    raise RuntimeError("cancelled")
                _set_job(
                    job_id,
                    progress=float(p),
                    message=msg,
                    status="running",
                    run_id=run_id,
                )

            def on_partial(
                ready: int,
                total: int,
                ready_indices: Optional[list[int]] = None,
            ) -> None:
                payload = _batch_payload(out_dir)
                _set_job(
                    job_id,
                    progress=max(0.05, float(ready) / float(max(total, 1))),
                    message=f"Depth {ready}/{total}…",
                    result=payload,
                    ready_frames=ready,
                    total_frames=total,
                    ready_indices=list(ready_indices or []),
                    run_id=run_id,
                )

            if kind == "preview":
                result_params = run_batch_preview(
                    out_dir,
                    params,
                    model_key=body.model_key,
                    max_size=body.max_size,
                    infer_backend=body.infer_backend,
                    on_progress=on_progress,
                    on_partial=on_partial,
                )
            else:
                result_params = run_batch(
                    out_dir,
                    params,
                    model_key=body.model_key,
                    max_size=body.max_size,
                    infer_backend=body.infer_backend,
                    folder_name=body.folder_name,
                    on_progress=on_progress,
                    on_partial=on_partial,
                )
            payload = _batch_payload(out_dir)
            if result_params.get("modal_billing"):
                payload["modal_billing"] = result_params["modal_billing"]
            _set_job(
                job_id,
                status="done",
                progress=1.0,
                message="Done",
                result=payload,
                run_id=run_id,
                error=None,
            )
            _end_infer(backend, state={"out_dir": str(out_dir)})
        except Exception as exc:
            err = str(exc)
            if err == "cancelled":
                _set_job(
                    job_id,
                    status="cancelled",
                    error="cancelled",
                    message="Cancelled",
                    run_id=run_id,
                )
                _end_infer(backend, state={"out_dir": str(out_dir)})
                return
            traceback.print_exc()
            _set_job(
                job_id,
                status="error",
                error=err,
                message=err,
                run_id=run_id,
            )
            _end_infer(backend, error=err)

    threading.Thread(target=worker, daemon=True).start()
    return job_id


@router.post("/batch/stills/prepare")
async def batch_stills_prepare(
    files: list[UploadFile] = File(...),
    names: str = Form("[]"),
    model_key: str = Form("depth-pro"),
    max_size: int = Form(756),
    infer_backend: Optional[str] = Form(None),
    folder_name: str = Form("depth"),
):
    """Import a folder of stills. First image is the look reference; no depth yet."""
    key = normalize_model_key(model_key)
    try:
        parsed = json.loads(names or "[]")
    except json.JSONDecodeError:
        parsed = []
    if not isinstance(parsed, list):
        parsed = []
    labels = [str(n) for n in parsed]
    tmp_root = Path(tempfile.mkdtemp(prefix="batch-stills-"))
    pairs: list[tuple[str, Path]] = []
    try:
        for i, upload in enumerate(files):
            label = labels[i] if i < len(labels) and labels[i] else (upload.filename or f"{i:06d}.jpg")
            suffix = Path(label).suffix.lower() or ".jpg"
            dest = tmp_root / f"{i:06d}{suffix}"
            with dest.open("wb") as handle:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            pairs.append((label, dest))
        state = prepare_batch(
            pairs,
            key,
            int(max_size),
            infer_backend=infer_backend,
            folder_name=folder_name,
        )
        return _batch_payload(Path(state["out_dir"]))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


@router.get("/batch/stills/{run_id}")
def batch_stills_get(run_id: str):
    return _batch_payload(_batch_dir(run_id))


@router.post("/batch/stills/{run_id}/preview")
def batch_stills_preview(run_id: str, body: BatchParamsBody = BatchParamsBody()):
    job_id = _start_batch_job(run_id, "preview", body)
    return {"job_id": job_id, "run_id": run_id}


@router.post("/batch/stills/{run_id}/run")
def batch_stills_run(run_id: str, body: BatchParamsBody = BatchParamsBody()):
    job_id = _start_batch_job(run_id, "run", body)
    return {"job_id": job_id, "run_id": run_id}


@router.post("/batch/stills/{run_id}/render")
def batch_stills_render(run_id: str, body: BatchParamsBody = BatchParamsBody()):
    out_dir = _batch_dir(run_id)
    params = normalize_film_params({**load_film_params(out_dir), **body.stack_params()})
    params["ema_enabled"] = False
    if (
        body.vis_min_m is not None
        or body.vis_max_m is not None
        or body.gamma is not None
        or body.cmap is not None
    ):
        params["bounds_locked"] = True
    save_film_params(out_dir, params)
    try:
        render_film_vis(out_dir, params)
        meta = load_frames_meta(out_dir) or {}
        n = int(meta.get("frame_count") or 0)
        vis_n = sum(
            1
            for i in range(n)
            if (out_dir / "frames" / f"{i:06d}" / "vis.png").is_file()
        )
        if n > 0 and vis_n >= n and int(meta.get("saved_count") or 0) > 0:
            export_batch_folder(out_dir, body.folder_name or meta.get("export_folder"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return _batch_payload(out_dir)


@router.get("/batch/stills/{run_id}/sample")
def batch_stills_sample(
    run_id: str,
    u: float = 0.5,
    v: float = 0.5,
    frame: int = 0,
):
    out_dir = _batch_dir(run_id)
    try:
        return sample_film_depth(out_dir, int(frame), float(u), float(v))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class BatchFolderBody(BaseModel):
    path: str = ""
    paths: list[str] = Field(default_factory=list)
    model_key: Optional[str] = None
    max_size: Optional[int] = None
    infer_backend: Optional[str] = None
    folder_name: Optional[str] = None


class BatchFoldersBody(BatchParamsBody):
    paths: list[str] = Field(default_factory=list)


_PICK_JS = r"""
ObjC.import("AppKit");
var panel = $.NSOpenPanel.openPanel;
panel.setCanChooseFiles(false);
panel.setCanChooseDirectories(true);
panel.setAllowsMultipleSelection(true);
panel.setCanCreateDirectories(false);
panel.setPrompt("Add");
panel.setMessage("Select folders. Command-click for more than one. A folder of clip folders is added as one batch per clip.");
var ok = panel.runModal;
if (ok != 1) {
  "";
} else {
  var urls = panel.URLs;
  var out = [];
  for (var i = 0; i < urls.count; i++) {
    out.push(ObjC.unwrap(urls.objectAtIndex(i).path));
  }
  out.join("\n");
}
"""


def _pick_source_folders() -> list[str]:
    proc = subprocess.run(
        ["osascript", "-l", "JavaScript"],
        input=_PICK_JS,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if "User canceled" in err or "-128" in err:
            return []
        # Older pickers: one folder.
        fallback = subprocess.run(
            ["osascript", "-e", 'POSIX path of (choose folder with prompt "Add image folders")'],
            capture_output=True,
            text=True,
        )
        if fallback.returncode != 0:
            err2 = (fallback.stderr or "").strip()
            if "User canceled" in err2 or "-128" in err2:
                return []
            raise RuntimeError(err2 or err or "Could not open the folder picker")
        path = (fallback.stdout or "").strip()
        return [str(Path(path).expanduser().resolve())] if path else []
    paths = []
    for line in (proc.stdout or "").splitlines():
        raw = line.strip()
        if raw:
            paths.append(str(Path(raw).expanduser().resolve()))
    return paths


def _expand_paths(paths: list[str]) -> dict[str, Any]:
    folders: list[dict[str, Any]] = []
    seen: set[str] = set()
    omitted = 0
    for raw in paths:
        found, skipped = discover_batches(raw)
        omitted += skipped
        for info in found:
            if info["path"] in seen:
                continue
            if len(folders) >= BATCH_MAX_FOLDERS:
                omitted += 1
                continue
            seen.add(info["path"])
            folders.append(info)
    if not folders:
        raise ValueError("No image folders found.")
    return {"folders": folders, "omitted_folders": omitted}


@router.post("/batch/stills/pick")
def batch_stills_pick():
    try:
        paths = _pick_source_folders()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    if not paths:
        return {"paths": [], "cancelled": True}
    return {"paths": paths, "cancelled": False}


@router.post("/batch/stills/inspect")
def batch_stills_inspect(body: BatchFolderBody):
    paths = [p.strip() for p in body.paths if p and p.strip()]
    if body.path and body.path.strip():
        paths.insert(0, body.path.strip())
    if not paths:
        raise HTTPException(status_code=400, detail="Give a folder path")
    try:
        return _expand_paths(paths)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/batch/stills/open")
def batch_stills_open(body: BatchFolderBody):
    """Load a folder from disk so plates can be written back into it."""
    key = normalize_model_key(body.model_key or "depth-pro")
    try:
        info = folder_summary(body.path)
        chosen, _omitted = images_in_folder(info["path"])
        state = prepare_batch(
            chosen,
            key,
            int(body.max_size or 756),
            infer_backend=body.infer_backend,
            folder_name=body.folder_name or "depth",
            source_dir=info["path"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return _batch_payload(Path(state["out_dir"]))


@router.post("/batch/stills/run-folders")
def batch_stills_run_folders(body: BatchFoldersBody):
    paths = [p.strip() for p in body.paths if p and p.strip()]
    if not paths:
        raise HTTPException(status_code=400, detail="Add at least one folder")
    job_id = uuid.uuid4().hex[:12]
    _set_job(
        job_id,
        status="running",
        progress=0.0,
        message="Queued folders…",
        kind="batch_folders",
        phase="run",
        run_id=None,
        result={"folders": []},
        error=None,
    )
    params = body.stack_params()
    backend = _begin_infer(body.infer_backend)

    def worker() -> None:
        try:

            def cancelled() -> bool:
                return bool((_get_job(job_id) or {}).get("cancel"))

            def on_update(frac: float, msg: str, folders: list[dict[str, Any]]) -> None:
                _set_job(
                    job_id,
                    progress=float(frac),
                    message=msg,
                    status="running",
                    result={"folders": folders},
                )

            summary = run_folders(
                paths,
                params,
                model_key=body.model_key or "depth-pro",
                max_size=int(body.max_size or 756),
                infer_backend=body.infer_backend,
                folder_name=body.folder_name or "depth",
                on_update=on_update,
                should_cancel=cancelled,
            )
            if summary.get("cancelled"):
                _set_job(
                    job_id,
                    status="cancelled",
                    progress=1.0,
                    message="Cancelled",
                    result=summary,
                    error="cancelled",
                )
                _end_infer(backend)
                return
            failed = int(summary.get("error_count") or 0)
            total = int(summary.get("folder_count") or 0)
            if failed and failed >= total:
                err = next(
                    (
                        f["error"]
                        for f in summary.get("folders") or []
                        if f.get("error")
                    ),
                    "Batch failed",
                )
                _set_job(
                    job_id,
                    status="error",
                    progress=1.0,
                    message=str(err),
                    result=summary,
                    error=str(err),
                )
                _end_infer(backend, error=str(err))
                return
            msg = f"Saved {summary.get('saved_count') or 0} images across {total} folders"
            if failed:
                msg += f" · {failed} failed"
            _set_job(
                job_id,
                status="done",
                progress=1.0,
                message=msg,
                result=summary,
                error=None,
            )
            _end_infer(backend)
        except Exception as exc:
            err = str(exc)
            if err == "cancelled":
                _set_job(
                    job_id,
                    status="cancelled",
                    message="Cancelled",
                    error="cancelled",
                )
                _end_infer(backend)
                return
            traceback.print_exc()
            _set_job(
                job_id,
                status="error",
                error=err,
                message=err,
            )
            _end_infer(backend, error=err)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}
