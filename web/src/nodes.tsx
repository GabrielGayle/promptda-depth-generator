import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useRef } from "react";
import { bust } from "./api";
import { AspectPreview } from "./AspectPreview";
import { lookDefaults, useKitchen } from "./kitchenContext";

function Shell({
  title,
  selected,
  collapsed,
  tuning,
  onToggle,
  onDelete,
  deletable,
  wide,
  children,
}: {
  title: string;
  selected?: boolean;
  collapsed?: boolean;
  tuning?: boolean;
  onToggle?: () => void;
  onDelete?: () => void;
  deletable?: boolean;
  wide?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      className={`dl-node${selected ? " selected" : ""}${
        collapsed ? " collapsed" : ""
      }${wide ? " wide" : ""}${tuning ? " tuning" : ""}`}
    >
      <div
        className="dl-title"
        onClick={(e) => {
          e.stopPropagation();
          onToggle?.();
        }}
        role={onToggle ? "button" : undefined}
      >
        <span>{title}</span>
        <span className="dl-title-actions">
          {deletable && onDelete ? (
            <button
              type="button"
              className="dl-delete nodrag nopan"
              title="Delete node"
              onClick={(e) => {
                e.stopPropagation();
                onDelete();
              }}
            >
              ×
            </button>
          ) : null}
          {onToggle ? (
            <span className="dl-chevron">{collapsed ? "▸" : "▾"}</span>
          ) : null}
        </span>
      </div>
      <div className="dl-body">{children}</div>
    </div>
  );
}

/** Low-res schematic: which depth percentile band maps into full display brightness. */
function WindowExplainer({
  lo,
  hi,
  gamma,
}: {
  lo: number;
  hi: number;
  gamma: number;
}) {
  const loClamped = Math.min(Math.max(lo, 0), 100);
  const hiClamped = Math.min(Math.max(Math.max(hi, loClamped + 1), 1), 100);
  const left = loClamped;
  const width = hiClamped - loClamped;

  const span = hiClamped - loClamped;
  const nearBlack = Math.max(0, loClamped - span);
  const cells: number[] = [];
  for (let i = 0; i < 48; i++) {
    const pct = (i / 47) * 100;
    let v: number;
    if (pct <= loClamped) {
      if (pct <= nearBlack || loClamped <= nearBlack) v = 0;
      else v = (pct - nearBlack) / (loClamped - nearBlack);
    } else if (pct >= hiClamped) {
      v = 0;
    } else {
      v = 1 - (pct - loClamped) / span;
    }
    v = Math.pow(Math.min(Math.max(v, 0), 1), gamma);
    cells.push(v);
  }

  return (
    <div className="window-explainer">
      <div className="window-axis-label">
        <span>Near</span>
        <span>depth %</span>
        <span>Far</span>
      </div>
      <div className="window-track" title="Depth percentiles across the scene">
        <div className="window-track-fill" />
        <div
          className="window-band"
          style={{ left: `${left}%`, width: `${width}%` }}
        />
        <div className="window-tick" style={{ left: `${loClamped}%` }} />
        <div className="window-tick" style={{ left: `${hiClamped}%` }} />
      </div>
      <div className="window-band-caption">
        Peak white at Near · black both sides
      </div>
      <div
        className="window-ramp"
        title="Remapped display (peak at Near window)"
      >
        {cells.map((v, i) => (
          <span
            key={i}
            style={{ background: `rgb(${v * 255}, ${v * 255}, ${v * 255})` }}
          />
        ))}
      </div>
      <div className="window-ramp-caption">
        Nearer → black · peak at lo · farther → black (γ {gamma.toFixed(2)})
      </div>
    </div>
  );
}

function stopNodeDrag(e: React.SyntheticEvent) {
  e.stopPropagation();
}

function NodeThumb({
  preview,
  evalToken,
  alt,
}: {
  preview?: string;
  evalToken: number;
  alt: string;
}) {
  if (!preview) return null;
  return (
    <AspectPreview
      src={bust(preview, evalToken)}
      alt={alt}
      className="dl-thumb"
      maxEdge={180}
    />
  );
}

export function DepthNode({ data, selected }: NodeProps) {
  const { rgbUrl, evalToken, openLightbox, inferBackend } = useKitchen();
  const d = data as any;
  const backend = d.infer_backend || inferBackend;
  const device = d.infer_device;
  const cost = d.modal_cost;
  return (
    <Shell title="Depth" selected={selected}>
      <Handle
        type="target"
        position={Position.Left}
        id="in"
        title="Are.na / source"
      />
      <div>
        run: {d.run_id || "—"}
        {backend ? (
          <span className="dl-backend">
            {" "}
            · {backend}
            {device ? `/${device}` : ""}
            {typeof cost === "number" ? ` · ~$${cost.toFixed(4)}` : ""}
          </span>
        ) : null}
      </div>
      {rgbUrl ? (
        <AspectPreview
          src={bust(rgbUrl, evalToken)}
          alt="Original"
          asButton
          caption="Original"
          maxEdge={160}
          onClick={(e) => {
            e.stopPropagation();
            openLightbox(bust(rgbUrl, evalToken), "Original");
          }}
        />
      ) : null}
      <Handle type="source" position={Position.Right} id="out" />
    </Shell>
  );
}

export function SmoothNode({ id, data, selected }: NodeProps) {
  const d = data as any;
  const { meta, evalToken, nodePreviews, patchNodeData, deleteNode } =
    useKitchen();
  const collapsed = Boolean(d.collapsed ?? true);
  const preview = nodePreviews[id];
  const smoothMeta = meta?.smooth;
  const defaults = smoothMeta?.defaults;

  const inferMode = (): string => {
    if (d.mode) return String(d.mode);
    const t = String(d.temporal ?? defaults?.temporal ?? "ema");
    const w = String(d.window_mode ?? defaults?.window_mode ?? "locked");
    if (t === "off" && w === "off") return "none";
    if (t === "flow") return "flow";
    if (t === "off") return "window";
    return "ema";
  };
  const mode = inferMode();
  const windowMode = String(
    mode === "none"
      ? "off"
      : d.window_mode ?? defaults?.window_mode ?? "locked"
  );
  const temporal =
    mode === "none"
      ? "off"
      : mode === "window"
        ? "off"
        : mode === "flow"
          ? "flow"
          : "ema";
  const emaAlpha = Number(d.ema_alpha ?? defaults?.ema_alpha ?? 0.45);
  const flowPrior = Number(
    d.flow_prior ??
      (d.flow_weight != null
        ? 1 - Number(d.flow_weight)
        : defaults?.flow_prior ?? 0.75)
  );
  const occThresh = Number(d.occ_thresh ?? defaults?.occ_thresh ?? 1.5);
  const lo = Number(d.lo_pct ?? defaults?.lo_pct ?? 2);
  const hi = Number(d.hi_pct ?? defaults?.hi_pct ?? 98);
  const splineInterval = Number(
    d.spline_interval ?? defaults?.spline_interval ?? 30
  );
  const windowLock = String(d.window_lock ?? defaults?.window_lock ?? "first");

  const modeLabel =
    smoothMeta?.modes?.find((m) => m.key === mode)?.label ||
    (mode === "none"
      ? "None (pass-through)"
      : mode === "window"
        ? "Window only"
        : mode === "flow"
          ? "Flow-guided"
          : "EMA temporal");

  const setMode = (next: string) => {
    if (next === "none") {
      patchNodeData(id, {
        mode: "none",
        temporal: "off",
        window_mode: "off",
      });
    } else if (next === "window") {
      patchNodeData(id, {
        mode: "window",
        temporal: "off",
        window_mode:
          windowMode === "off" ? "locked" : windowMode || "locked",
      });
    } else if (next === "flow") {
      patchNodeData(id, {
        mode: "flow",
        temporal: "flow",
        window_mode: windowMode === "off" ? "locked" : windowMode,
      });
    } else {
      patchNodeData(id, {
        mode: "ema",
        temporal: "ema",
        window_mode: windowMode === "off" ? "locked" : windowMode,
      });
    }
  };

  return (
    <Shell
      title="Smooth"
      selected={selected}
      collapsed={collapsed}
      onToggle={() => patchNodeData(id, { collapsed: !collapsed })}
      deletable
      onDelete={() => deleteNode(id)}
    >
      <Handle type="target" position={Position.Left} id="in" title="Depth" />
      <div
        className="dl-controls nodrag nopan nowheel"
        onPointerDown={stopNodeDrag}
        onClick={stopNodeDrag}
      >
        <div className="dl-summary">{modeLabel}</div>
        <p className="dl-hint">
          Optional stabilizer — set to None to pass depth through unchanged
        </p>
        {!collapsed ? (
          <>
            <label>
              Mode
              <select
                className="nodrag nopan"
                value={mode}
                onPointerDown={stopNodeDrag}
                onChange={(e) => setMode(e.target.value)}
              >
                {(
                  smoothMeta?.modes || [
                    { key: "none", label: "None (pass-through)" },
                    { key: "ema", label: "EMA temporal" },
                    { key: "flow", label: "Flow-guided" },
                    { key: "window", label: "Window only" },
                  ]
                ).map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
            {mode === "none" ? (
              <p className="dl-hint">
                Pass-through: no temporal blend, no display-window lock.
              </p>
            ) : (
              <>
                <label>
                  Display window
                  <select
                    className="nodrag nopan"
                    value={windowMode}
                    onPointerDown={stopNodeDrag}
                    onChange={(e) =>
                      patchNodeData(id, {
                        window_mode: e.target.value,
                        mode:
                          temporal === "off"
                            ? e.target.value === "off"
                              ? "none"
                              : "window"
                            : mode,
                      })
                    }
                  >
                    {(smoothMeta?.window_modes || []).map((m) => (
                      <option key={m.key} value={m.key}>
                        {m.label}
                      </option>
                    ))}
                    {!smoothMeta?.window_modes?.length ? (
                      <>
                        <option value="off">Per-frame (raw)</option>
                        <option value="locked">Locked window</option>
                        <option value="spline">Spline-smoothed window</option>
                      </>
                    ) : null}
                  </select>
                </label>
                {windowMode === "locked" ? (
                  <label>
                    Lock source
                    <select
                      className="nodrag nopan"
                      value={windowLock}
                      onPointerDown={stopNodeDrag}
                      onChange={(e) =>
                        patchNodeData(id, { window_lock: e.target.value })
                      }
                    >
                      {(smoothMeta?.window_locks || []).map((m) => (
                        <option key={m.key} value={m.key}>
                          {m.label}
                        </option>
                      ))}
                      {!smoothMeta?.window_locks?.length ? (
                        <>
                          <option value="first">First frame</option>
                          <option value="median">Median of clip</option>
                        </>
                      ) : null}
                    </select>
                  </label>
                ) : null}
                {windowMode === "spline" ? (
                  <label>
                    Spline interval ({splineInterval})
                    <input
                      className="nodrag nopan nowheel"
                      type="range"
                      min={4}
                      max={60}
                      step={1}
                      value={splineInterval}
                      onPointerDown={stopNodeDrag}
                      onChange={(e) =>
                        patchNodeData(id, {
                          spline_interval: parseInt(e.target.value, 10),
                        })
                      }
                    />
                  </label>
                ) : null}
                {windowMode !== "off" ? (
                  <>
                    <label>
                      Near % ({lo.toFixed(0)})
                      <input
                        className="nodrag nopan nowheel"
                        type="range"
                        min={1}
                        max={40}
                        step={0.5}
                        value={lo}
                        onPointerDown={stopNodeDrag}
                        onChange={(e) =>
                          patchNodeData(id, {
                            lo_pct: parseFloat(e.target.value),
                          })
                        }
                      />
                    </label>
                    <label>
                      Far % ({hi.toFixed(0)})
                      <input
                        className="nodrag nopan nowheel"
                        type="range"
                        min={20}
                        max={99}
                        step={0.5}
                        value={hi}
                        onPointerDown={stopNodeDrag}
                        onChange={(e) =>
                          patchNodeData(id, {
                            hi_pct: parseFloat(e.target.value),
                          })
                        }
                      />
                    </label>
                  </>
                ) : null}
                {mode === "ema" || mode === "flow" ? (
                  <>
                    {mode === "ema" ? (
                      <label>
                        Current weight ({emaAlpha.toFixed(2)})
                        <input
                          className="nodrag nopan nowheel"
                          type="range"
                          min={0.1}
                          max={1}
                          step={0.05}
                          value={emaAlpha}
                          onPointerDown={stopNodeDrag}
                          onChange={(e) =>
                            patchNodeData(id, {
                              ema_alpha: parseFloat(e.target.value),
                            })
                          }
                        />
                      </label>
                    ) : null}
                    {mode === "flow" ? (
                      <>
                        <label>
                          Prior weight ({flowPrior.toFixed(2)})
                          <input
                            className="nodrag nopan nowheel"
                            type="range"
                            min={0.1}
                            max={0.95}
                            step={0.05}
                            value={flowPrior}
                            onPointerDown={stopNodeDrag}
                            onChange={(e) =>
                              patchNodeData(id, {
                                flow_prior: parseFloat(e.target.value),
                              })
                            }
                          />
                        </label>
                        <label>
                          Occlusion thresh ({occThresh.toFixed(1)} px)
                          <input
                            className="nodrag nopan nowheel"
                            type="range"
                            min={0.5}
                            max={5}
                            step={0.25}
                            value={occThresh}
                            onPointerDown={stopNodeDrag}
                            onChange={(e) =>
                              patchNodeData(id, {
                                occ_thresh: parseFloat(e.target.value),
                              })
                            }
                          />
                        </label>
                        <p className="dl-hint">
                          Farneback + forward–backward mask; occlusions use
                          current depth.
                        </p>
                      </>
                    ) : null}
                  </>
                ) : null}
              </>
            )}
          </>
        ) : null}
        <NodeThumb preview={preview} evalToken={evalToken} alt="smooth" />
      </div>
      <Handle type="source" position={Position.Right} id="out" />
    </Shell>
  );
}

export function FreqFuseNode({ id, data, selected }: NodeProps) {
  const d = data as any;
  const { meta, evalToken, nodePreviews, patchNodeData, deleteNode } =
    useKitchen();
  const collapsed = Boolean(d.collapsed ?? true);
  const preview = nodePreviews[id];
  const defaults = meta?.freq_fuse?.defaults;
  const beta = Number(d.beta ?? defaults?.beta ?? 1);
  const blurSigma = Number(d.blur_sigma ?? defaults?.blur_sigma ?? 4);
  const backendLabel =
    String(d.backend_label || meta?.freq_fuse?.backend_label || "stub EMA");

  return (
    <Shell
      title="Freq Fuse"
      selected={selected}
      collapsed={collapsed}
      onToggle={() => patchNodeData(id, { collapsed: !collapsed })}
      deletable
      onDelete={() => deleteNode(id)}
    >
      <Handle type="target" position={Position.Left} id="in" title="Depth" />
      <div
        className="dl-controls nodrag nopan nowheel"
        onPointerDown={stopNodeDrag}
        onClick={stopNodeDrag}
      >
        <div className="dl-summary">
          β {beta.toFixed(2)} · σ {blurSigma.toFixed(1)}
        </div>
        <p className="dl-hint">
          Video low-freq + Depth Pro edges · {backendLabel}
        </p>
        {!collapsed ? (
          <>
            <label>
              Detail β ({beta.toFixed(2)})
              <input
                className="nodrag nopan nowheel"
                type="range"
                min={0}
                max={2}
                step={0.05}
                value={beta}
                onPointerDown={stopNodeDrag}
                onChange={(e) =>
                  patchNodeData(id, { beta: parseFloat(e.target.value) })
                }
              />
            </label>
            <label>
              Blur σ ({blurSigma.toFixed(1)})
              <input
                className="nodrag nopan nowheel"
                type="range"
                min={0.5}
                max={16}
                step={0.5}
                value={blurSigma}
                onPointerDown={stopNodeDrag}
                onChange={(e) =>
                  patchNodeData(id, {
                    blur_sigma: parseFloat(e.target.value),
                  })
                }
              />
            </label>
          </>
        ) : null}
        <NodeThumb preview={preview} evalToken={evalToken} alt="freq fuse" />
      </div>
      <Handle type="source" position={Position.Right} id="out" />
    </Shell>
  );
}

export function ArenaNode({ id, data, selected }: NodeProps) {
  const d = data as any;
  const {
    busy,
    progress,
    patchNodeUi,
    deleteNode,
    openLightbox,
    resolveArenaNode,
    loadArenaIntoDepth,
  } = useKitchen();
  const linkValue = String(d.channel_url || d.url || "");
  const status = String(d.status || "idle");
  const isChannel = Boolean(
    d.channel_url || d.kind === "channel" || d.channel
  );
  const hasPreview = Boolean(d.block?.thumb_url || d.block?.image_url);
  const previewReady = hasPreview && status === "ready";
  const thumb = d.block?.thumb_url || d.block?.image_url;
  const title = d.block?.title || d.channel?.title || "Are.na";
  const resolving = status === "resolving" || progress === "upload";
  const loading = status === "loading" || progress === "infer";
  const disabled = busy || resolving || loading;

  return (
    <Shell
      title="Are.na"
      selected={selected}
      deletable
      onDelete={() => deleteNode(id)}
      wide
    >
      <div
        className="dl-controls nodrag nopan nowheel arena-node"
        onPointerDown={stopNodeDrag}
        onClick={stopNodeDrag}
      >
        <label>
          Link
          <input
            className="nodrag nopan"
            type="text"
            placeholder="are.na/block/… or channel"
            value={linkValue}
            disabled={disabled}
            onPointerDown={stopNodeDrag}
            onChange={(e) =>
              patchNodeUi(id, {
                url: e.target.value,
                channel_url: null,
                channel: null,
                block: null,
                kind: null,
                imageable_count: null,
                status: "idle",
                error: null,
              })
            }
            onKeyDown={(e) => {
              if (e.key === "Enter" && linkValue.trim()) {
                e.preventDefault();
                void resolveArenaNode(id, "auto");
              }
            }}
          />
        </label>

        <div className="arena-actions">
          <button
            type="button"
            className="nodrag nopan"
            disabled={disabled || !linkValue.trim()}
            onClick={() => void resolveArenaNode(id, "auto")}
          >
            {resolving ? "…" : "Fetch"}
          </button>
          <button
            type="button"
            className={`nodrag nopan${previewReady ? " primary" : ""}`}
            disabled={disabled || (!linkValue.trim() && !hasPreview)}
            onClick={() => void loadArenaIntoDepth(id, "auto")}
            title={
              hasPreview
                ? "Process the previewed image through depth"
                : "Download block image and run depth"
            }
          >
            {loading ? "Loading…" : "→ Depth"}
          </button>
        </div>

        {isChannel ? (
          <div className="arena-actions">
            <button
              type="button"
              className="nodrag nopan"
              disabled={disabled || !linkValue.trim()}
              onClick={() => void resolveArenaNode(id, "shuffle")}
              title="Pick a random imageable block — preview only"
            >
              {resolving ? "…" : "Shuffle"}
            </button>
            <button
              type="button"
              className="nodrag nopan"
              disabled
              title="Use Are.na Out → Roll for shuffle + randomize + save"
            >
              Batch
            </button>
          </div>
        ) : null}

        {d.error ? <p className="dl-error">{d.error}</p> : null}

        {d.block || d.channel ? (
          <div className="arena-meta">
            {isChannel && d.channel ? (
              <div className="dl-summary">
                Channel · {d.channel.title}
                {d.imageable_count != null
                  ? ` · ${d.imageable_count} images`
                  : ""}
              </div>
            ) : (
              <div className="dl-summary">Block · {d.block?.class}</div>
            )}
            <div className="dl-summary arena-block-title" title={title}>
              {title}
            </div>
            {thumb ? (
              <AspectPreview
                src={thumb}
                alt={title}
                asButton
                caption={d.block?.id != null ? `#${d.block.id}` : undefined}
                maxEdge={220}
                onClick={(e) => {
                  e.stopPropagation();
                  openLightbox(thumb, title);
                }}
              />
            ) : null}
            {previewReady ? (
              <p className="dl-hint arena-preview-hint">
                Like it? Click <strong>→ Depth</strong> to run the pipeline.
                {isChannel ? " Shuffle again to try another." : ""}
              </p>
            ) : null}
          </div>
        ) : (
          <p className="dl-hint">
            Paste a block or channel link. Fetch or Shuffle previews first —
            then → Depth to process.
          </p>
        )}
      </div>
      <Handle type="source" position={Position.Right} id="out" />
    </Shell>
  );
}

export function LookNode({ id, data, selected }: NodeProps) {
  const d = data as any;
  const {
    meta,
    evalToken,
    rgbUrl,
    runId,
    nodePreviews,
    windowTune,
    patchNodeData,
    deleteNode,
    startWindowTune,
  } = useKitchen();
  const collapsed = Boolean(d.collapsed ?? true);
  const preset = d.preset || "ledger";
  const label = meta?.looks.find((l) => l.key === preset)?.label || preset;
  const preview = nodePreviews[id];
  const lo = Number(d.lo_pct ?? 2);
  const hi = Number(d.hi_pct ?? 98);
  const gamma = Number(d.gamma ?? 1);
  const looks = meta?.looks || [];
  const tuning = windowTune?.nodeId === id;
  const canTune = Boolean(runId && rgbUrl);

  const setPreset = (key: string) => {
    const defaults = lookDefaults(meta, key);
    const patch: Record<string, unknown> = {
      preset: key,
      lo_pct: defaults?.lo_pct ?? 2,
      hi_pct: defaults?.hi_pct ?? 98,
      gamma: defaults?.gamma ?? 1,
      cmap: "inherit",
    };
    if (defaults?.cmap === "wash" || defaults?.blur_amount != null) {
      patch.blur_amount = defaults?.blur_amount ?? 0.4;
    }
    patchNodeData(id, patch);
  };

  const effectiveCmap =
    d.cmap && d.cmap !== "inherit"
      ? d.cmap
      : lookDefaults(meta, preset)?.cmap || "gray";
  const showWashBlur = effectiveCmap === "wash";
  const blurAmount = Number(
    d.blur_amount ?? lookDefaults(meta, preset)?.blur_amount ?? 0.4
  );

  return (
    <Shell
      title={`Look · ${label}`}
      selected={selected}
      collapsed={collapsed}
      tuning={tuning}
      onToggle={() => patchNodeData(id, { collapsed: !collapsed })}
      deletable
      onDelete={() => deleteNode(id)}
    >
      <Handle type="target" position={Position.Left} id="in" title="Input" />
      <div
        className="dl-controls nodrag nopan nowheel"
        onPointerDown={stopNodeDrag}
        onClick={stopNodeDrag}
      >
        <label>
          Preset
          <select
            className="nodrag nopan"
            value={preset}
            disabled={looks.length === 0}
            onPointerDown={stopNodeDrag}
            onChange={(e) => setPreset(e.target.value)}
          >
            {looks.length === 0 ? (
              <option value={preset}>{preset}</option>
            ) : (
              looks.map((l) => (
                <option key={l.key} value={l.key}>
                  {l.label}
                </option>
              ))
            )}
          </select>
        </label>
        <p className="dl-hint">Blend lives on the incoming wire — click it</p>
        {!collapsed ? (
          <>
            <WindowExplainer lo={lo} hi={hi} gamma={gamma} />
            <label>
              Near window ({lo.toFixed(0)})
              <input
                className="nodrag nopan nowheel"
                type="range"
                min={1}
                max={40}
                step={0.5}
                value={lo}
                onPointerDown={stopNodeDrag}
                onChange={(e) =>
                  patchNodeData(id, { lo_pct: parseFloat(e.target.value) })
                }
              />
            </label>
            <label>
              Far window ({hi.toFixed(0)})
              <input
                className="nodrag nopan nowheel"
                type="range"
                min={20}
                max={99}
                step={0.5}
                value={hi}
                onPointerDown={stopNodeDrag}
                onChange={(e) =>
                  patchNodeData(id, { hi_pct: parseFloat(e.target.value) })
                }
              />
            </label>
            <label>
              Gamma ({gamma.toFixed(2)})
              <input
                className="nodrag nopan nowheel"
                type="range"
                min={0.4}
                max={1.8}
                step={0.01}
                value={gamma}
                onPointerDown={stopNodeDrag}
                onChange={(e) =>
                  patchNodeData(id, { gamma: parseFloat(e.target.value) })
                }
              />
            </label>
            <label>
              Colormap
              <select
                className="nodrag nopan"
                value={d.cmap || "inherit"}
                onPointerDown={stopNodeDrag}
                onChange={(e) => {
                  const next = e.target.value;
                  const patch: Record<string, unknown> = { cmap: next };
                  if (
                    next === "wash" ||
                    (next === "inherit" &&
                      lookDefaults(meta, preset)?.cmap === "wash")
                  ) {
                    if (d.blur_amount == null) {
                      patch.blur_amount =
                        lookDefaults(meta, preset)?.blur_amount ?? 0.4;
                    }
                  }
                  patchNodeData(id, patch);
                }}
              >
                <option value="inherit">Preset default</option>
                {(meta?.cmaps || []).map((c) => (
                  <option key={c} value={c}>
                    {c === "photo"
                      ? "Photo (from image)"
                      : c === "wash"
                        ? "Wash (from image)"
                        : c}
                  </option>
                ))}
              </select>
            </label>
            {showWashBlur ? (
              <label>
                Wash blur ({blurAmount.toFixed(2)})
                <input
                  className="nodrag nopan nowheel"
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={blurAmount}
                  onPointerDown={stopNodeDrag}
                  onChange={(e) =>
                    patchNodeData(id, {
                      blur_amount: parseFloat(e.target.value),
                    })
                  }
                />
              </label>
            ) : null}
          </>
        ) : null}
        <button
          type="button"
          className="dl-tune-btn nodrag nopan"
          disabled={!canTune}
          title={
            canTune
              ? "Click the scene to park near/far windows"
              : "Load an image first"
          }
          onPointerDown={stopNodeDrag}
          onClick={(e) => {
            e.stopPropagation();
            startWindowTune(id);
          }}
        >
          Tune windows
        </button>
        <NodeThumb preview={preview} evalToken={evalToken} alt={preset} />
      </div>
      <Handle type="source" position={Position.Right} id="out" />
    </Shell>
  );
}

export function SceneNode({ id, data, selected }: NodeProps) {
  const d = data as any;
  const { evalToken, nodePreviews, patchNodeData, deleteNode } = useKitchen();
  const collapsed = Boolean(d.collapsed ?? true);
  const preview = nodePreviews[id];

  return (
    <Shell
      title="Scene"
      selected={selected}
      collapsed={collapsed}
      onToggle={() => patchNodeData(id, { collapsed: !collapsed })}
      deletable
      onDelete={() => deleteNode(id)}
    >
      <Handle type="target" position={Position.Left} id="in" title="Input" />
      <div
        className="dl-controls nodrag nopan nowheel"
        onPointerDown={stopNodeDrag}
        onClick={stopNodeDrag}
      >
        <div className="dl-summary">
          Matte: {d.mode || "smart"} · p {Number(d.protect ?? 0).toFixed(2)}
        </div>
        <p className="dl-hint">Blend lives on the incoming wire — click it</p>
        <NodeThumb preview={preview} evalToken={evalToken} alt="scene" />
      </div>
      <Handle type="source" position={Position.Right} id="out" />
    </Shell>
  );
}

export function ReliefNode({ id, data, selected }: NodeProps) {
  const d = data as any;
  const { meta, evalToken, nodePreviews, patchNodeData, deleteNode } =
    useKitchen();
  const collapsed = Boolean(d.collapsed ?? true);
  const preview = nodePreviews[id];
  const gc = Number(d.gc_alpha ?? meta?.relief?.default_gc_alpha ?? 10);
  const invert = Boolean(d.invert ?? meta?.relief?.default_invert ?? false);
  const maxSide = Number(
    d.max_side ?? meta?.relief?.default_max_side ?? 512
  );
  const sides = meta?.relief?.max_sides || [384, 512, 768, 1024];

  return (
    <Shell
      title="Relief"
      selected={selected}
      collapsed={collapsed}
      onToggle={() => patchNodeData(id, { collapsed: !collapsed })}
      deletable
      onDelete={() => deleteNode(id)}
    >
      <Handle type="target" position={Position.Left} id="in" title="Input" />
      <div
        className="dl-controls nodrag nopan nowheel"
        onPointerDown={stopNodeDrag}
        onClick={stopNodeDrag}
      >
        <div className="dl-summary">
          α {gc.toFixed(1)} · {maxSide}px{invert ? " · invert" : ""}
        </div>
        <p className="dl-hint">
          Juices the incoming plate into a smooth bas-relief
        </p>
        {!collapsed ? (
          <>
            <label>
              Compression α ({gc.toFixed(1)})
              <input
                className="nodrag nopan nowheel"
                type="range"
                min={1}
                max={40}
                step={0.5}
                value={gc}
                onPointerDown={stopNodeDrag}
                onChange={(e) =>
                  patchNodeData(id, { gc_alpha: parseFloat(e.target.value) })
                }
              />
            </label>
            <label>
              Solve size
              <select
                className="nodrag nopan"
                value={maxSide}
                onPointerDown={stopNodeDrag}
                onChange={(e) =>
                  patchNodeData(id, { max_side: parseInt(e.target.value, 10) })
                }
              >
                {sides.map((s) => (
                  <option key={s} value={s}>
                    {s}px
                  </option>
                ))}
              </select>
            </label>
            <label>
              <input
                className="nodrag nopan"
                type="checkbox"
                checked={invert}
                onPointerDown={stopNodeDrag}
                onChange={(e) =>
                  patchNodeData(id, { invert: e.target.checked })
                }
              />{" "}
              Invert (bright = far)
            </label>
          </>
        ) : null}
        <NodeThumb preview={preview} evalToken={evalToken} alt="relief" />
      </div>
      <Handle type="source" position={Position.Right} id="out" />
    </Shell>
  );
}

export function BlendNode({ id, data, selected }: NodeProps) {
  const d = data as any;
  const { meta, evalToken, nodePreviews, patchNodeData, deleteNode } =
    useKitchen();
  const preview = nodePreviews[id];
  const mode = d.mode || "normal";
  const opacity = Number(d.opacity ?? 1);
  const modes = meta?.blend_modes || [];

  return (
    <Shell
      title="Blend"
      selected={selected}
      deletable
      onDelete={() => deleteNode(id)}
    >
      <Handle
        type="target"
        position={Position.Left}
        id="base"
        style={{ top: "35%" }}
      />
      <Handle
        type="target"
        position={Position.Left}
        id="overlay"
        style={{ top: "70%" }}
      />
      <div
        className="dl-controls nodrag nopan nowheel"
        onPointerDown={stopNodeDrag}
        onClick={stopNodeDrag}
      >
        <label>
          Mode
          <select
            className="nodrag nopan"
            value={mode}
            disabled={modes.length === 0}
            onPointerDown={stopNodeDrag}
            onChange={(e) => patchNodeData(id, { mode: e.target.value })}
          >
            {modes.length === 0 ? (
              <option value={mode}>{mode}</option>
            ) : (
              modes.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))
            )}
          </select>
        </label>
        <label>
          Opacity ({opacity.toFixed(2)})
          <input
            className="nodrag nopan nowheel"
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={opacity}
            onPointerDown={stopNodeDrag}
            onChange={(e) =>
              patchNodeData(id, { opacity: parseFloat(e.target.value) })
            }
          />
        </label>
        <NodeThumb preview={preview} evalToken={evalToken} alt="blend" />
      </div>
      <Handle type="source" position={Position.Right} id="out" />
    </Shell>
  );
}

export function OutputNode({ id, data, selected }: NodeProps) {
  const d = data as any;
  const {
    outputUrl,
    evalToken,
    progress,
    patchNodeUi,
    openLightbox,
    frameCount,
    runId,
  } = useKitchen();
  const width = Math.min(720, Math.max(240, Number(d.preview_width ?? 420)));
  const isVideo = frameCount > 1;
  const src = outputUrl ? bust(outputUrl, evalToken) : null;
  const npzHref =
    runId && src ? bust(`/api/runs/${runId}/export/npz`, evalToken) : null;
  const strataHref =
    runId && src ? bust(`/api/runs/${runId}/export/strata`, evalToken) : null;

  return (
    <Shell title="Output" selected={selected} wide>
      <Handle type="target" position={Position.Left} id="in" />
      <div
        className="dl-output nodrag nopan nowheel"
        style={{ width }}
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="dl-output-toolbar">
          <button
            type="button"
            title="Smaller"
            disabled={width <= 240}
            onClick={() =>
              patchNodeUi(id, { preview_width: Math.max(240, width - 80) })
            }
          >
            −
          </button>
          <button
            type="button"
            title="Larger"
            disabled={width >= 720}
            onClick={() =>
              patchNodeUi(id, { preview_width: Math.min(720, width + 80) })
            }
          >
            +
          </button>
          <button
            type="button"
            title="Enlarge"
            disabled={!src}
            onClick={() => src && openLightbox(src, "Output")}
          >
            Expand
          </button>
          {src && !isVideo ? (
            <a href={src} download className="dl-output-dl">
              Download
            </a>
          ) : null}
          {npzHref && !isVideo ? (
            <a
              href={npzHref}
              download
              className="dl-output-dl"
              title="Float32 depth (m), inv_smooth, valid mask, rgb + output plates"
            >
              NPZ
            </a>
          ) : null}
          {strataHref && !isVideo ? (
            <a
              href={strataHref}
              download
              className="dl-output-dl"
              title="Grey bas-relief PNG for Strata: near = dark, background = white"
            >
              Strata
            </a>
          ) : null}
        </div>

        {isVideo ? (
          <p className="dl-hint">
            Video kitchens use <strong>Video Out</strong> for scrub / Export MP4.
          </p>
        ) : null}

        <div className={`dl-output-frame${progress ? " busy" : ""}`}>
          {src ? (
            <AspectPreview
              src={src}
              alt="Output"
              className={progress ? "dimmed" : ""}
            />
          ) : (
            <div className="preview-empty">No output yet</div>
          )}
          {progress === "evaluate" ? (
            <div className="progress-overlay compact">
              <div className="progress-bar" />
            </div>
          ) : null}
        </div>
      </div>
    </Shell>
  );
}

export function VideoOutNode({ id, data, selected }: NodeProps) {
  const d = data as any;
  const {
    outputUrl,
    evalToken,
    progress,
    patchNodeUi,
    openLightbox,
    frameCount,
    readyFrames,
    readyIndices,
    frameIndex,
    sampleFps,
    inferStatus,
    inferMode,
    inferBackend,
    previewGifUrl,
    frameUrls,
    exportUrl,
    playing,
    setPlaying,
    scrubFrame,
    exportVideo,
    busy,
    maxSize,
    setMaxSize,
    meta,
    frameWidth,
    frameHeight,
    frameStatusExpanded,
    setFrameStatusExpanded,
    inferVideoRun,
  } = useKitchen();
  const width = Math.min(720, Math.max(240, Number(d.preview_width ?? 420)));
  const baked = frameUrls.length > 0 && inferStatus === "done";
  const currentBaked =
    baked && frameUrls[frameIndex]
      ? bust(frameUrls[frameIndex], evalToken)
      : null;
  const stillSrc = outputUrl ? bust(outputUrl, evalToken) : null;
  const gifSrc = previewGifUrl ? bust(previewGifUrl, evalToken) : null;
  const src = currentBaked || stillSrc || gifSrc;
  const scrubMax = Math.max(1, Math.min(readyFrames, frameCount));
  const canExport =
    !busy &&
    inferStatus === "done" &&
    readyFrames >= frameCount &&
    frameCount > 1;
  const canInfer =
    frameCount > 1 &&
    (inferStatus === "pending" ||
      inferStatus === "ready" ||
      inferStatus === "done" ||
      inferStatus === "error") &&
    !busy &&
    progress !== "infer";
  const inferring = progress === "infer" || inferStatus === "running";
  const maxSizes = meta?.max_sizes || [518, 756, 1008, 1536];
  const maxRes = Math.max(...maxSizes);
  const readySet = new Set(readyIndices);
  const modeLabel =
    inferMode === "modal_parallel"
      ? "Modal · parallel"
      : inferMode === "local_sequential"
        ? "Local · sequential"
        : inferBackend === "modal"
          ? "Modal · parallel"
          : "Local · sequential";
  const resLabel =
    frameWidth && frameHeight
      ? `${frameWidth}×${frameHeight}`
      : `max ${maxSize}`;

  return (
    <Shell title="Video Out" selected={selected} wide>
      <Handle type="target" position={Position.Left} id="in" />
      <div
        className="dl-output nodrag nopan nowheel"
        style={{ width }}
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="dl-output-toolbar">
          <button
            type="button"
            title="Smaller"
            disabled={width <= 240}
            onClick={() =>
              patchNodeUi(id, { preview_width: Math.max(240, width - 80) })
            }
          >
            −
          </button>
          <button
            type="button"
            title="Larger"
            disabled={width >= 720}
            onClick={() =>
              patchNodeUi(id, { preview_width: Math.min(720, width + 80) })
            }
          >
            +
          </button>
          <button
            type="button"
            title="Enlarge"
            disabled={!src}
            onClick={() => src && openLightbox(src, "Video Out")}
          >
            Expand
          </button>
          <button
            type="button"
            className={`nodrag nopan${canInfer ? " primary" : ""}`}
            disabled={!canInfer || inferring}
            title="Run Depth Pro on prepared frames"
            onClick={() => void inferVideoRun()}
          >
            {inferring ? "Inferring…" : "Infer"}
          </button>
          {canExport ? (
            <button
              type="button"
              className="nodrag nopan"
              disabled={busy}
              onClick={() => void exportVideo()}
            >
              Export MP4
            </button>
          ) : null}
          {exportUrl ? (
            <a
              href={bust(exportUrl, evalToken)}
              download
              className="dl-output-dl"
            >
              Download MP4
            </a>
          ) : null}
        </div>

        <div className="video-out-res nodrag nopan">
          <label title="Longest edge cap for Depth Pro (re-extracts on Infer if changed)">
            Max
            <select
              value={maxSize}
              disabled={busy || inferring}
              onChange={(e) => setMaxSize(parseInt(e.target.value, 10))}
            >
              {maxSizes.map((n) => (
                <option key={n} value={n}>
                  {n}
                  {n === maxRes ? " (max)" : ""}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="nodrag nopan"
            disabled={busy || inferring || maxSize === maxRes}
            title={`Use Depth Pro native max (${maxRes})`}
            onClick={() => setMaxSize(maxRes)}
          >
            Max res
          </button>
          <span className="dl-hint" title="Frames are resized so the longest side ≤ Max">
            {resLabel}
          </span>
        </div>

        <div className="dl-output-video-meta">
          {baked
            ? `Frame ${frameIndex + 1} / ${frameCount} · ${sampleFps.toFixed(1)} fps · ${modeLabel}`
            : inferStatus === "running"
              ? `${modeLabel} · ${readyFrames}/${frameCount}`
              : inferStatus === "pending"
                ? `Prepared · ${frameCount} frames · ${resLabel} · Infer here`
                : frameCount > 1
                  ? `Video · ${frameCount} frames · ${resLabel}`
                  : "Wire a look plate here"}
        </div>

        <div className={`dl-output-frame${progress ? " busy" : ""}`}>
          {src ? (
            <AspectPreview
              src={src}
              alt="Video Out"
              className={progress ? "dimmed" : ""}
            />
          ) : (
            <div className="preview-empty">No output yet</div>
          )}
          {progress === "evaluate" ? (
            <div className="progress-overlay compact">
              <div className="progress-bar" />
            </div>
          ) : null}
        </div>

        {frameCount > 1 ? (
          <div className="frame-status nodrag nopan">
            <button
              type="button"
              className="frame-status-toggle"
              onClick={() => setFrameStatusExpanded(!frameStatusExpanded)}
            >
              {frameStatusExpanded ? "▾" : "▸"} Frame status · {readyFrames}/
              {frameCount}
              {inferMode ? ` · ${modeLabel}` : ""}
            </button>
            {frameStatusExpanded ? (
              <div className="frame-status-grid" title={modeLabel}>
                {Array.from({ length: frameCount }, (_, i) => {
                  let st = "pending";
                  if (readySet.has(i) || (inferStatus === "done" && i < readyFrames)) {
                    st = "ready";
                  } else if (
                    inferStatus === "running" &&
                    inferMode === "local_sequential" &&
                    i === readyFrames
                  ) {
                    st = "running";
                  } else if (inferStatus === "running" && !readySet.has(i)) {
                    st = "queued";
                  }
                  return (
                    <button
                      key={i}
                      type="button"
                      className={`frame-chip ${st}${
                        i === frameIndex ? " active" : ""
                      }`}
                      title={`Frame ${i + 1}: ${st}`}
                      onClick={() => {
                        if (st === "ready" || baked) {
                          setPlaying(false);
                          scrubFrame(i);
                        }
                      }}
                    >
                      {i + 1}
                    </button>
                  );
                })}
              </div>
            ) : null}
          </div>
        ) : null}

        {baked ? (
          <>
            <div className="dl-filmstrip">
              {frameUrls.map((url, i) => (
                <button
                  key={url}
                  type="button"
                  className={`dl-filmstrip-thumb${
                    i === frameIndex ? " active" : ""
                  }`}
                  onClick={() => {
                    setPlaying(false);
                    scrubFrame(i);
                  }}
                  title={`Frame ${i + 1}`}
                >
                  <img src={bust(url, evalToken)} alt="" />
                </button>
              ))}
            </div>
            <div className="dl-output-scrub">
              <input
                type="range"
                min={0}
                max={Math.max(0, scrubMax - 1)}
                step={1}
                value={Math.min(frameIndex, Math.max(0, scrubMax - 1))}
                onChange={(e) => {
                  setPlaying(false);
                  scrubFrame(parseInt(e.target.value, 10));
                }}
              />
              <button
                type="button"
                onClick={() => setPlaying(!playing)}
                disabled={scrubMax <= 1}
              >
                {playing ? "Pause" : "Play"}
              </button>
            </div>
          </>
        ) : null}
      </div>
    </Shell>
  );
}

export function VideoInNode({ id, data, selected }: NodeProps) {
  const d = data as any;
  const {
    busy,
    progress,
    deleteNode,
    prepareVideoNode,
    evalToken,
    maxSize,
    frameWidth,
    frameHeight,
  } = useKitchen();
  const fileRef = useRef<HTMLInputElement>(null);
  const status = String(d.status || "idle");
  const sourceUrl = d.source_url as string | undefined;
  const previewGif = d.preview_gif_url as string | undefined;
  const filename = String(d.filename || "");
  const frameCount = Number(d.frame_count || 0);
  const fw = Number(d.frame_width || frameWidth || 0);
  const fh = Number(d.frame_height || frameHeight || 0);
  const preparing = status === "preparing" || progress === "upload";
  const disabled = busy || preparing;
  const gifSrc = previewGif ? bust(previewGif, evalToken) : null;

  return (
    <Shell
      title="Video In"
      selected={selected}
      deletable
      onDelete={() => deleteNode(id)}
      wide
    >
      <div
        className="dl-controls nodrag nopan nowheel arena-node video-in-node"
        onPointerDown={stopNodeDrag}
        onClick={stopNodeDrag}
      >
        <input
          ref={fileRef}
          type="file"
          accept="video/mp4,video/quicktime,video/webm,video/*"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void prepareVideoNode(id, f);
            e.target.value = "";
          }}
        />
        <div className="arena-actions">
          <button
            type="button"
            className="nodrag nopan"
            disabled={disabled}
            onClick={() => fileRef.current?.click()}
          >
            {preparing
              ? "Preparing…"
              : sourceUrl || previewGif
                ? "Replace…"
                : "Load video"}
          </button>
        </div>
        {filename ? (
          <div className="dl-summary" title={filename}>
            {filename}
            {frameCount > 0 ? ` · ${frameCount} frames` : ""}
            {fw && fh ? ` · ${fw}×${fh}` : ""}
            {` · max ${d.max_size ?? maxSize}`}
          </div>
        ) : (
          <p className="dl-hint">
            Load a video here, arrange the kitchen, then Infer on{" "}
            <strong>Video Out</strong>.
          </p>
        )}
        {d.error ? <p className="dl-error">{d.error}</p> : null}
        {gifSrc ? (
          <div className="video-in-preview nodrag nopan">
            <img
              src={gifSrc}
              alt="Source preview"
              className="video-in-gif"
              draggable={false}
              onPointerDown={stopNodeDrag}
            />
            <p className="dl-hint arena-preview-hint">
              Short RGB preview (not depth) · longest edge ≤ Max
            </p>
          </div>
        ) : null}
        {status === "pending" || status === "ready" ? (
          <p className="dl-hint arena-preview-hint">
            Prepared — click <strong>Infer</strong> on Video Out to depth the
            clip.
          </p>
        ) : null}
      </div>
      <Handle type="source" position={Position.Right} id="out" />
    </Shell>
  );
}

export function ArenaOutNode({ id, data, selected }: NodeProps) {
  const d = data as any;
  const {
    busy,
    runId,
    arenaChannels,
    arenaAuth,
    patchNodeUi,
    deleteNode,
    refreshArenaChannels,
    publishArenaOut,
    runArenaBatch,
    stopArenaBatch,
  } = useKitchen();
  const channelId = String(d.channel_id || "");
  const title = String(d.title || "");
  const status = String(d.status || "idle");
  const publishing = status === "publishing";
  const batching = status === "batching";
  const batchCount = Math.max(1, Math.min(200, Number(d.batch_count ?? 5)));
  const wackiness = Math.min(1, Math.max(0, Number(d.wackiness ?? 0.45)));
  const maxFidelity = Boolean(d.max_fidelity);
  const batchI = Number(d.batch_i || 0);
  const batchN = Number(d.batch_n || batchCount);
  const batchProgress = Math.min(1, Math.max(0, Number(d.batch_progress || 0)));
  const batchMessage = String(d.batch_message || "");
  const batchDownloaded = Number(d.batch_downloaded || 0);
  const batchInferred = Number(d.batch_inferred || 0);
  const batchCached = Number(d.batch_cached || 0);
  const batchCooked = Number(d.batch_cooked || 0);
  const batchPublished = Number(d.batch_published || 0);
  const batchItems = (Array.isArray(d.batch_items) ? d.batch_items : []) as {
    index: number;
    status: string;
    title?: string | null;
    arena_url?: string | null;
    error?: string | null;
  }[];
  const lastUrl = d.last_block_url as string | undefined;
  const disabled = busy || publishing || batching;
  const showPanel = batching || batchItems.length > 0;

  return (
    <Shell
      title="Are.na Out"
      selected={selected}
      deletable
      onDelete={() => deleteNode(id)}
      wide
    >
      <Handle type="target" position={Position.Left} id="in" title="Composite" />
      <div
        className="dl-controls nodrag nopan nowheel arena-node"
        onPointerDown={stopNodeDrag}
        onClick={stopNodeDrag}
      >
        {!arenaAuth ? (
          <p className="dl-error">
            Set ARENA_ACCESS_TOKEN in .env and restart the API.
          </p>
        ) : null}
        <label>
          Channel
          <select
            className="nodrag nopan"
            value={channelId}
            disabled={disabled || !arenaAuth}
            onPointerDown={stopNodeDrag}
            onChange={(e) =>
              patchNodeUi(id, { channel_id: e.target.value, error: null })
            }
            onFocus={() => void refreshArenaChannels()}
          >
            <option value="">Recent channels…</option>
            {arenaChannels.map((c) => (
              <option key={c.id} value={String(c.slug || c.id)}>
                {c.title}
                {c.visibility === "private" ? " · private" : ""}
              </option>
            ))}
          </select>
        </label>
        <label>
          Title
          <input
            className="nodrag nopan"
            type="text"
            placeholder="Depth Ledger"
            value={title}
            disabled={disabled}
            onPointerDown={stopNodeDrag}
            onChange={(e) => patchNodeUi(id, { title: e.target.value })}
          />
        </label>
        <div className="arena-actions">
          <button
            type="button"
            className="nodrag nopan"
            disabled={disabled || !arenaAuth}
            onClick={() => void refreshArenaChannels()}
          >
            Refresh
          </button>
          <button
            type="button"
            className="nodrag nopan primary"
            disabled={disabled || !arenaAuth || !runId || !channelId}
            onClick={() => void publishArenaOut(id)}
            title="Upload current composite to Are.na"
          >
            {publishing ? "Saving…" : "Save to Are.na"}
          </button>
        </div>

        <div className="arena-batch">
          <label>
            Rolls ({batchCount})
            <input
              className="nodrag nopan nowheel"
              type="range"
              min={1}
              max={200}
              step={1}
              value={batchCount}
              disabled={disabled}
              onPointerDown={stopNodeDrag}
              onChange={(e) =>
                patchNodeUi(id, { batch_count: parseInt(e.target.value, 10) })
              }
            />
          </label>
          <label>
            Wackiness ({Math.round(wackiness * 100)}%)
            <input
              className="nodrag nopan nowheel"
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={wackiness}
              disabled={disabled}
              onPointerDown={stopNodeDrag}
              onChange={(e) =>
                patchNodeUi(id, { wackiness: parseFloat(e.target.value) })
              }
            />
          </label>
          <label>
            <input
              className="nodrag nopan"
              type="checkbox"
              checked={maxFidelity}
              disabled={disabled}
              onPointerDown={stopNodeDrag}
              onChange={(e) =>
                patchNodeUi(id, { max_fidelity: e.target.checked })
              }
            />{" "}
            Max fidelity
          </label>
          <div className="arena-actions">
            {batching ? (
              <button
                type="button"
                className="nodrag nopan danger"
                onClick={() => stopArenaBatch()}
              >
                Stop · {batchI}/{batchN}
              </button>
            ) : (
              <button
                type="button"
                className="nodrag nopan primary"
                disabled={!arenaAuth || !channelId || busy}
                onClick={() => void runArenaBatch(id)}
                title="Unique Are.na stills → randomize kitchen → save"
              >
                Roll ×{batchCount}
              </button>
            )}
          </div>
          <p className="dl-hint">
            Each roll picks a unique Are.na still (no repeats in the set) from
            the channel + nested subchannels, sampling ~50 at a time and
            refilling as needed → randomizes looks by wackiness → saves to
            Are.na. Repeat blocks reuse the local infer cache. Download, infer,
            and cook overlap. Max fidelity pins infer + Relief to the largest
            sizes (keeps wackiness).
          </p>
        </div>

        {showPanel ? (
          <div className={`arena-batch-panel${batching ? " live" : ""}`}>
            <div className="arena-batch-panel-head">
              <strong>{batching ? "In progress" : "Last batch"}</strong>
              <span>
                {batchPublished}/{batchN} saved
              </span>
            </div>
            <div
              className="arena-batch-meter"
              role="progressbar"
              aria-valuenow={Math.round(batchProgress * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div
                className="arena-batch-meter-fill"
                style={{ width: `${Math.round(batchProgress * 100)}%` }}
              />
            </div>
            {batchMessage ? (
              <p className="arena-batch-msg">{batchMessage}</p>
            ) : null}
            <div className="arena-batch-counts">
              <span>↓ {batchDownloaded}</span>
              <span>GPU {batchInferred}</span>
              {batchCached > 0 ? <span>cache {batchCached}</span> : null}
              <span>✎ {batchCooked}</span>
              <span>↑ {batchPublished}</span>
            </div>
            {batchItems.length ? (
              <ul className="arena-batch-items">
                {batchItems.map((it) => (
                  <li
                    key={it.index}
                    className={`arena-batch-item status-${it.status}`}
                  >
                    <span className="arena-batch-item-idx">#{it.index}</span>
                    <span className="arena-batch-item-status">{it.status}</span>
                    <span className="arena-batch-item-title">
                      {it.error
                        ? it.error
                        : it.title ||
                          (it.status === "queued" ? "waiting" : "—")}
                    </span>
                    {it.arena_url ? (
                      <a
                        href={it.arena_url}
                        target="_blank"
                        rel="noreferrer"
                        className="arena-batch-item-link"
                        onClick={(e) => e.stopPropagation()}
                      >
                        ↗
                      </a>
                    ) : null}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}

        {d.error ? <p className="dl-error">{d.error}</p> : null}
        {lastUrl ? (
          <a
            className="dl-hint arena-out-link"
            href={lastUrl}
            target="_blank"
            rel="noreferrer"
            onClick={(e) => e.stopPropagation()}
          >
            Open last block ↗
          </a>
        ) : null}
      </div>
    </Shell>
  );
}

export const nodeTypes = {
  depth: DepthNode,
  arena: ArenaNode,
  arena_out: ArenaOutNode,
  video_in: VideoInNode,
  video_out: VideoOutNode,
  smooth: SmoothNode,
  freq_fuse: FreqFuseNode,
  look: LookNode,
  scene: SceneNode,
  relief: ReliefNode,
  blend: BlendNode,
  output: OutputNode,
};
