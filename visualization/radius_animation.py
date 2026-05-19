"""Render learned-radius flocking traces to MP4 animations."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle


_POSITION_KEYS = (
    "pre_action_positions",
    "positions",
    "agent_positions",
)
_STATE_KEYS = (
    "pre_action_agent_states",
    "agent_states",
    "states",
    "state_history",
    "agent_states_hist",
)
_RADIUS_KEYS = (
    "physical_radii",
    "radius_values",
    "radii",
    "radii_physical",
    "radius_physical",
    "selected_radii",
)
_NORMALIZED_RADIUS_KEYS = (
    "radius_actions",
    "normalized_radius_actions",
    "normalized_radii",
    "actions",
    "action",
)
_MASK_KEYS = (
    "active_masks",
    "active_mask",
    "padding_masks",
    "padding_mask",
)
_HEADING_KEYS = (
    "headings",
    "absolute_headings",
)
_VELOCITY_KEYS = (
    "velocities",
    "agent_velocities",
)


@dataclass(frozen=True)
class _TraceView:
    positions: np.ndarray
    radii: np.ndarray
    active_masks: np.ndarray
    headings: Optional[np.ndarray]
    returns: Optional[np.ndarray]
    spatial_entropy: Optional[np.ndarray]
    velocity_entropy: Optional[np.ndarray]
    entropy: Optional[np.ndarray]
    steps: np.ndarray


def render_radius_animation(trace: Any, output_path: os.PathLike[str] | str, fps: int = 20, dpi: int = 150) -> str:
    """Render a radius-action rollout trace to an MP4 file.

    The renderer expects positions and radii aligned by decision step: frame
    ``t`` should contain the positions observed before action ``t`` and the
    physical radii selected for that same state. Accepted trace formats are:

    - ``{"frames": [...]}``, where each frame has positions or agent states,
      physical radii or normalized radius actions, optional masks, headings,
      reward/return, and entropy fields.
    - Array-style dictionaries/objects with ``positions`` or ``agent_states``
      shaped ``(T, N, 2+)`` plus ``physical_radii``/``radius_values``/``radii``
      shaped ``(T, N)``. If only normalized actions are present, provide
      ``radius_min`` and ``radius_max`` at top level or in ``metadata``.

    Args:
        trace: In-memory trace object or dictionary.
        output_path: Destination MP4 path.
        fps: Frames per second.
        dpi: Figure DPI used while encoding.

    Returns:
        The output path as a string.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("matplotlib FFMpegWriter is unavailable; install ffmpeg to render MP4 files")

    view = _normalize_trace(trace)
    output_path = os.fspath(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    xlim, ylim = _axis_limits(view.positions, view.radii, view.active_masks)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    colors = _agent_colors(view.positions.shape[1])
    inactive_offsets = np.full((view.positions.shape[1], 2), np.nan, dtype=float)
    scatter = ax.scatter(
        inactive_offsets[:, 0],
        inactive_offsets[:, 1],
        s=34,
        c=colors,
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
    )

    circles = []
    for color in colors:
        face_color = (color[0], color[1], color[2], 0.08)
        edge_color = (color[0], color[1], color[2], 0.48)
        circle = Circle(
            (0.0, 0.0),
            0.0,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=1.1,
            visible=False,
            zorder=2,
        )
        ax.add_patch(circle)
        circles.append(circle)

    arrow_length = 0.035 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    quiver = None
    if view.headings is not None:
        quiver = ax.quiver(
            inactive_offsets[:, 0],
            inactive_offsets[:, 1],
            np.zeros(view.positions.shape[1]),
            np.zeros(view.positions.shape[1]),
            color=colors,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
            headwidth=3.0,
            headlength=4.0,
            headaxislength=3.5,
            zorder=5,
        )

    def update(frame_idx: int):
        positions = view.positions[frame_idx]
        radii = view.radii[frame_idx]
        active = view.active_masks[frame_idx] & np.isfinite(positions).all(axis=1) & np.isfinite(radii)

        offsets = inactive_offsets.copy()
        offsets[active] = positions[active]
        scatter.set_offsets(offsets)

        for agent_idx, circle in enumerate(circles):
            is_visible = bool(active[agent_idx])
            circle.set_visible(is_visible)
            if is_visible:
                circle.center = tuple(positions[agent_idx])
                circle.radius = max(float(radii[agent_idx]), 0.0)

        artists = [scatter, *circles]
        if quiver is not None:
            heading = view.headings[frame_idx]
            u = np.zeros(view.positions.shape[1], dtype=float)
            v = np.zeros(view.positions.shape[1], dtype=float)
            valid_heading = active & np.isfinite(heading)
            u[valid_heading] = np.cos(heading[valid_heading]) * arrow_length
            v[valid_heading] = np.sin(heading[valid_heading]) * arrow_length
            quiver.set_offsets(offsets)
            quiver.set_UVC(u, v)
            artists.append(quiver)

        ax.set_title(_status_text(view, frame_idx), fontsize=10)
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=view.positions.shape[0],
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        metadata={"artist": "neighbor_selection_rl_flocking"},
        extra_args=["-pix_fmt", "yuv420p"],
    )
    try:
        anim.save(output_path, writer=writer, dpi=dpi)
    finally:
        plt.close(fig)
    return output_path


def _normalize_trace(trace: Any) -> _TraceView:
    frames = _get_field(trace, ("frames", "records", "samples"))
    if frames is None:
        step_records = _get_field(trace, ("steps",))
        if step_records is not None:
            step_records = list(step_records)
            if step_records and isinstance(step_records[0], Mapping):
                frames = step_records
    if frames is not None:
        frames = list(frames)
        if frames:
            return _normalize_frame_trace(trace, frames)
    return _normalize_array_trace(trace)


def _normalize_frame_trace(trace: Any, frames: Sequence[Any]) -> _TraceView:
    metadata = _get_field(trace, ("metadata",), default={}) or {}
    radius_min = _optional_scalar(_get_field(trace, ("radius_min",), default=_get_field(metadata, ("radius_min",))))
    radius_max = _optional_scalar(_get_field(trace, ("radius_max",), default=_get_field(metadata, ("radius_max",))))

    positions_seq = []
    radii_seq = []
    masks_seq = []
    headings_seq = []
    returns = []
    rewards = []
    steps = []
    spatial_entropy = []
    velocity_entropy = []
    entropy = []
    has_heading = False
    has_return = False
    has_reward = False
    has_spatial_entropy = False
    has_velocity_entropy = False
    has_entropy = False

    for frame_idx, frame in enumerate(frames):
        frame_info = _get_field(frame, ("info",), default={}) or {}
        state = _get_field(frame, ("pre_action_state", "state"), default={}) or {}

        positions, state_array = _extract_frame_positions(frame, state)
        n_agents = positions.shape[0]
        positions_seq.append(positions)

        radius_values = _extract_frame_radii(frame, frame_info, n_agents, radius_min, radius_max)
        radii_seq.append(radius_values)

        mask = _extract_frame_mask(frame, state, n_agents)
        masks_seq.append(mask)

        heading = _extract_frame_heading(frame, state, state_array, n_agents)
        headings_seq.append(heading)
        has_heading = has_heading or heading is not None

        step_value = _get_field(frame, ("step", "time_step", "t"), default=frame_idx)
        steps.append(int(step_value) if _is_finite_scalar(step_value) else frame_idx)

        return_value = _get_field(
            frame,
            ("return_so_far", "cumulative_return", "episode_return_so_far", "return"),
            default=_get_field(frame_info, ("return_so_far", "cumulative_return", "episode_return_so_far", "return")),
        )
        returns.append(_optional_scalar(return_value))
        has_return = has_return or returns[-1] is not None

        reward_value = _get_field(frame, ("reward",), default=_get_field(frame_info, ("reward",)))
        rewards.append(_optional_scalar(reward_value))
        has_reward = has_reward or rewards[-1] is not None

        sp_value = _get_field(frame, ("spatial_entropy",), default=_get_field(frame_info, ("spatial_entropy",)))
        spatial_entropy.append(_optional_scalar(sp_value))
        has_spatial_entropy = has_spatial_entropy or spatial_entropy[-1] is not None

        vel_value = _get_field(frame, ("velocity_entropy",), default=_get_field(frame_info, ("velocity_entropy",)))
        velocity_entropy.append(_optional_scalar(vel_value))
        has_velocity_entropy = has_velocity_entropy or velocity_entropy[-1] is not None

        ent_value = _get_field(frame, ("entropy",), default=_get_field(frame_info, ("entropy",)))
        entropy.append(_optional_scalar(ent_value))
        has_entropy = has_entropy or entropy[-1] is not None

    positions = _pad_agent_arrays(positions_seq, trailing_shape=(2,), fill_value=np.nan)
    radii = _pad_agent_arrays(radii_seq, trailing_shape=(), fill_value=np.nan)
    active_masks = _pad_agent_arrays(masks_seq, trailing_shape=(), fill_value=False).astype(bool)
    headings = None
    if has_heading:
        headings = _pad_optional_agent_arrays(headings_seq, positions.shape[1])

    return _TraceView(
        positions=positions,
        radii=radii,
        active_masks=active_masks,
        headings=headings,
        returns=_series_or_cumsum(returns, rewards, has_return, has_reward),
        spatial_entropy=_optional_series(spatial_entropy, has_spatial_entropy),
        velocity_entropy=_optional_series(velocity_entropy, has_velocity_entropy),
        entropy=_optional_series(entropy, has_entropy),
        steps=np.asarray(steps, dtype=int),
    )


def _normalize_array_trace(trace: Any) -> _TraceView:
    positions, states = _extract_array_positions(trace)
    radii = _extract_array_radii(trace, positions.shape[:2])
    active_masks = _extract_array_masks(trace, positions, radii)
    headings = _extract_array_headings(trace, states, positions.shape[:2])
    returns = _extract_array_returns(trace, positions.shape[0])
    spatial_entropy = _extract_array_series(trace, ("spatial_entropy",), positions.shape[0])
    velocity_entropy = _extract_array_series(trace, ("velocity_entropy",), positions.shape[0])
    entropy = _extract_array_series(trace, ("entropy",), positions.shape[0])
    steps = _extract_array_steps(trace, positions.shape[0])

    return _TraceView(
        positions=positions,
        radii=radii,
        active_masks=active_masks,
        headings=headings,
        returns=returns,
        spatial_entropy=spatial_entropy,
        velocity_entropy=velocity_entropy,
        entropy=entropy,
        steps=steps,
    )


def _extract_frame_positions(frame: Any, state: Any) -> tuple[np.ndarray, Optional[np.ndarray]]:
    positions = _get_field(frame, _POSITION_KEYS)
    state_array = None
    if positions is None:
        state_array = _get_field(frame, _STATE_KEYS)
        if state_array is None:
            state_array = _get_field(state, _STATE_KEYS)
        if state_array is None:
            raise ValueError("Each trace frame must contain positions or agent_states")
        state_array = np.asarray(state_array, dtype=float)
        positions = state_array[:, :2]
    positions = _ensure_frame_positions(positions)
    if state_array is None:
        state_array = _get_field(frame, _STATE_KEYS)
        if state_array is None:
            state_array = _get_field(state, _STATE_KEYS)
        state_array = None if state_array is None else np.asarray(state_array, dtype=float)
    return positions, state_array


def _extract_frame_radii(
    frame: Any,
    frame_info: Any,
    n_agents: int,
    radius_min: Optional[float],
    radius_max: Optional[float],
) -> np.ndarray:
    radii = _get_field(frame, _RADIUS_KEYS)
    if radii is None:
        radii = _get_field(frame_info, _RADIUS_KEYS)
    if radii is not None:
        return _ensure_frame_agent_values(radii, n_agents, "radius values")

    normalized = _get_field(frame, _NORMALIZED_RADIUS_KEYS)
    if normalized is None:
        normalized = _get_field(frame_info, _NORMALIZED_RADIUS_KEYS)
    if normalized is None:
        raise ValueError("Each trace frame must contain physical radii or normalized radius actions")
    frame_radius_min = _optional_scalar(_get_field(frame, ("radius_min",), default=_get_field(frame_info, ("radius_min",))))
    frame_radius_max = _optional_scalar(_get_field(frame, ("radius_max",), default=_get_field(frame_info, ("radius_max",))))
    radius_min = radius_min if frame_radius_min is None else frame_radius_min
    radius_max = radius_max if frame_radius_max is None else frame_radius_max
    return _normalized_to_physical(
        _ensure_frame_agent_values(normalized, n_agents, "normalized radius actions"),
        radius_min,
        radius_max,
    )


def _extract_frame_mask(frame: Any, state: Any, n_agents: int) -> np.ndarray:
    mask = _get_field(frame, _MASK_KEYS)
    if mask is None:
        mask = _get_field(state, _MASK_KEYS)
    if mask is None:
        return np.ones(n_agents, dtype=bool)
    return _ensure_frame_agent_values(mask, n_agents, "active mask").astype(bool)


def _extract_frame_heading(
    frame: Any,
    state: Any,
    state_array: Optional[np.ndarray],
    n_agents: int,
) -> Optional[np.ndarray]:
    heading = _get_field(frame, _HEADING_KEYS)
    if heading is None:
        heading = _get_field(state, _HEADING_KEYS)
    if heading is not None:
        return _ensure_frame_agent_values(heading, n_agents, "headings")
    if state_array is not None and state_array.ndim == 2 and state_array.shape[1] >= 5:
        return _ensure_frame_agent_values(state_array[:, 4], n_agents, "headings")
    velocity = _get_field(frame, _VELOCITY_KEYS)
    if velocity is None:
        velocity = _get_field(state, _VELOCITY_KEYS)
    if velocity is None and state_array is not None and state_array.ndim == 2 and state_array.shape[1] >= 4:
        velocity = state_array[:, 2:4]
    if velocity is None:
        return None
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != (n_agents, 2):
        return None
    return np.arctan2(velocity[:, 1], velocity[:, 0])


def _extract_array_positions(trace: Any) -> tuple[np.ndarray, Optional[np.ndarray]]:
    positions = _get_field(trace, _POSITION_KEYS)
    states = None
    if positions is None:
        states = _get_field(trace, _STATE_KEYS)
        if states is None:
            raise ValueError("Trace must contain positions or agent_states")
        states = _ensure_time_agent_state(states)
        positions = states[:, :, :2]
    else:
        positions = _ensure_time_agent_positions(positions)
        states = _get_field(trace, _STATE_KEYS)
        states = None if states is None else _ensure_time_agent_state(states)
    return positions, states


def _extract_array_radii(trace: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    radii = _get_field(trace, _RADIUS_KEYS)
    if radii is not None:
        return _ensure_time_agent_values(radii, expected_shape, "radius values")

    normalized = _get_field(trace, _NORMALIZED_RADIUS_KEYS)
    if normalized is None:
        raise ValueError("Trace must contain physical radii or normalized radius actions")

    metadata = _get_field(trace, ("metadata",), default={}) or {}
    radius_min = _optional_scalar(_get_field(trace, ("radius_min",), default=_get_field(metadata, ("radius_min",))))
    radius_max = _optional_scalar(_get_field(trace, ("radius_max",), default=_get_field(metadata, ("radius_max",))))
    return _normalized_to_physical(
        _ensure_time_agent_values(normalized, expected_shape, "normalized radius actions"),
        radius_min,
        radius_max,
    )


def _extract_array_masks(trace: Any, positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
    mask = _get_field(trace, _MASK_KEYS)
    if mask is None:
        return np.isfinite(positions).all(axis=2) & np.isfinite(radii)
    mask = np.asarray(mask)
    if mask.ndim == 1:
        if mask.shape[0] != positions.shape[1]:
            raise ValueError(f"active mask has shape {mask.shape}; expected ({positions.shape[1]},)")
        mask = np.broadcast_to(mask.astype(bool), positions.shape[:2])
    elif mask.ndim == 2:
        if mask.shape != positions.shape[:2]:
            raise ValueError(f"active mask has shape {mask.shape}; expected {positions.shape[:2]}")
        mask = mask.astype(bool)
    else:
        raise ValueError("active mask must be shaped (N,) or (T, N)")
    return mask


def _extract_array_headings(
    trace: Any,
    states: Optional[np.ndarray],
    expected_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    heading = _get_field(trace, _HEADING_KEYS)
    if heading is not None:
        return _ensure_time_agent_values(heading, expected_shape, "headings")
    if states is not None and states.shape[2] >= 5:
        return _ensure_time_agent_values(states[:, :, 4], expected_shape, "headings")
    velocity = _get_field(trace, _VELOCITY_KEYS)
    if velocity is None and states is not None and states.shape[2] >= 4:
        velocity = states[:, :, 2:4]
    if velocity is None:
        return None
    velocity = np.asarray(velocity, dtype=float)
    if velocity.ndim == 2 and velocity.shape == (expected_shape[1], 2):
        velocity = np.broadcast_to(velocity, (*expected_shape, 2))
    if velocity.shape != (*expected_shape, 2):
        return None
    return np.arctan2(velocity[:, :, 1], velocity[:, :, 0])


def _extract_array_returns(trace: Any, n_frames: int) -> Optional[np.ndarray]:
    returns = _extract_array_series(
        trace,
        ("return_so_far", "returns", "cumulative_returns", "episode_returns_so_far"),
        n_frames,
    )
    if returns is not None:
        return returns
    rewards = _extract_array_series(trace, ("rewards", "reward"), n_frames)
    if rewards is None:
        return None
    return np.cumsum(rewards)


def _extract_array_series(trace: Any, keys: tuple[str, ...], n_frames: int) -> Optional[np.ndarray]:
    values = _get_field(trace, keys)
    if values is None:
        infos = _get_field(trace, ("infos", "info"))
        if infos is not None and not isinstance(infos, Mapping):
            infos = list(infos)
            if len(infos) == n_frames:
                values = [
                    _optional_scalar(_get_field(info, keys))
                    for info in infos
                ]
    if values is None:
        return None
    return _ensure_time_series(values, n_frames, keys[0])


def _extract_array_steps(trace: Any, n_frames: int) -> np.ndarray:
    steps = _get_field(trace, ("steps", "time_steps", "t"))
    if steps is None:
        return np.arange(n_frames, dtype=int)
    steps = np.asarray(steps)
    if steps.ndim != 1 or steps.shape[0] != n_frames:
        raise ValueError(f"steps has shape {steps.shape}; expected ({n_frames},)")
    return steps.astype(int)


def _axis_limits(positions: np.ndarray, radii: np.ndarray, active_masks: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    active_positions = positions[active_masks & np.isfinite(positions).all(axis=2)]
    active_radii = radii[active_masks & np.isfinite(radii)]
    if active_positions.size == 0:
        raise ValueError("Trace contains no active finite positions")

    min_xy = np.nanmin(active_positions, axis=0)
    max_xy = np.nanmax(active_positions, axis=0)
    max_radius = float(np.nanmax(active_radii)) if active_radii.size else 0.0
    span = np.maximum(max_xy - min_xy, 1.0)
    margin = max(max_radius, 0.08 * float(np.max(span)), 1.0)
    center = 0.5 * (min_xy + max_xy)
    half_extent = 0.5 * np.maximum(span[0], span[1]) + margin
    xlim = (float(center[0] - half_extent), float(center[0] + half_extent))
    ylim = (float(center[1] - half_extent), float(center[1] + half_extent))
    return xlim, ylim


def _status_text(view: _TraceView, frame_idx: int) -> str:
    active = view.active_masks[frame_idx] & np.isfinite(view.radii[frame_idx])
    radius_mean = float(np.nanmean(view.radii[frame_idx][active])) if np.any(active) else np.nan
    parts = [
        f"step {int(view.steps[frame_idx])}",
        f"{frame_idx + 1}/{view.positions.shape[0]}",
    ]
    if view.returns is not None and np.isfinite(view.returns[frame_idx]):
        parts.append(f"return {view.returns[frame_idx]:.3g}")
    if np.isfinite(radius_mean):
        parts.append(f"radius mean {radius_mean:.3g}")
    entropy_parts = []
    if view.spatial_entropy is not None and np.isfinite(view.spatial_entropy[frame_idx]):
        entropy_parts.append(f"p={view.spatial_entropy[frame_idx]:.3g}")
    if view.velocity_entropy is not None and np.isfinite(view.velocity_entropy[frame_idx]):
        entropy_parts.append(f"v={view.velocity_entropy[frame_idx]:.3g}")
    if entropy_parts:
        parts.append("entropy " + ", ".join(entropy_parts))
    elif view.entropy is not None and np.isfinite(view.entropy[frame_idx]):
        parts.append(f"entropy {view.entropy[frame_idx]:.3g}")
    return " | ".join(parts)


def _agent_colors(n_agents: int) -> np.ndarray:
    if n_agents <= 20:
        cmap = plt.get_cmap("tab20")
        return np.asarray([cmap(i % 20) for i in range(n_agents)])
    cmap = plt.get_cmap("hsv")
    return np.asarray([cmap(i / max(n_agents, 1)) for i in range(n_agents)])


def _get_field(source: Any, names: Sequence[str], default: Any = None) -> Any:
    if source is None:
        return default
    for name in names:
        if isinstance(source, Mapping):
            if name in source and source[name] is not None:
                return source[name]
        elif hasattr(source, name):
            value = getattr(source, name)
            if value is not None:
                return value
    return default


def _ensure_frame_positions(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"positions must be shaped (N, 2+); got {array.shape}")
    return array[:, :2]


def _ensure_time_agent_positions(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3 or array.shape[2] < 2:
        raise ValueError(f"positions must be shaped (T, N, 2+) or (N, 2+); got {array.shape}")
    return array[:, :, :2]


def _ensure_time_agent_state(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3 or array.shape[2] < 2:
        raise ValueError(f"agent states must be shaped (T, N, D) or (N, D); got {array.shape}")
    return array


def _ensure_frame_agent_values(values: Any, n_agents: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape[0] == 1 and n_agents != 1:
        array = np.full(n_agents, float(array[0]), dtype=float)
    if array.shape[0] != n_agents:
        raise ValueError(f"{name} has shape {array.shape}; expected ({n_agents},)")
    return array


def _ensure_time_agent_values(values: Any, expected_shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(expected_shape, float(array), dtype=float)
    elif array.ndim == 1:
        if array.shape[0] == expected_shape[1]:
            array = np.broadcast_to(array[np.newaxis, :], expected_shape).astype(float)
        elif expected_shape[1] == 1 and array.shape[0] == expected_shape[0]:
            array = array[:, np.newaxis]
        else:
            raise ValueError(f"{name} has shape {array.shape}; expected {expected_shape}")
    if array.shape != expected_shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {expected_shape}")
    return array.astype(float)


def _ensure_time_series(values: Any, n_frames: int, name: str) -> np.ndarray:
    array = np.asarray([
        np.nan if value is None else value
        for value in np.asarray(values, dtype=object).reshape(-1)
    ], dtype=float)
    if array.shape[0] != n_frames:
        raise ValueError(f"{name} has shape {array.shape}; expected ({n_frames},)")
    return array


def _pad_agent_arrays(values: Sequence[Any], trailing_shape: tuple[int, ...], fill_value: Any) -> np.ndarray:
    n_frames = len(values)
    max_agents = max(np.asarray(value).shape[0] for value in values)
    output = np.full((n_frames, max_agents, *trailing_shape), fill_value)
    for frame_idx, value in enumerate(values):
        array = np.asarray(value)
        output[frame_idx, : array.shape[0], ...] = array
    return output


def _pad_optional_agent_arrays(values: Sequence[Optional[np.ndarray]], max_agents: int) -> np.ndarray:
    output = np.full((len(values), max_agents), np.nan, dtype=float)
    for frame_idx, value in enumerate(values):
        if value is not None:
            output[frame_idx, : value.shape[0]] = value
    return output


def _optional_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        return None
    scalar = float(array.reshape(-1)[0])
    return scalar if np.isfinite(scalar) else None


def _is_finite_scalar(value: Any) -> bool:
    scalar = _optional_scalar(value)
    return scalar is not None


def _optional_series(values: Sequence[Optional[float]], has_values: bool) -> Optional[np.ndarray]:
    if not has_values:
        return None
    return np.asarray([np.nan if value is None else value for value in values], dtype=float)


def _series_or_cumsum(
    returns: Sequence[Optional[float]],
    rewards: Sequence[Optional[float]],
    has_return: bool,
    has_reward: bool,
) -> Optional[np.ndarray]:
    if has_return:
        return _optional_series(returns, True)
    if not has_reward:
        return None
    rewards_array = np.asarray([0.0 if value is None else value for value in rewards], dtype=float)
    return np.cumsum(rewards_array)


def _normalized_to_physical(values: np.ndarray, radius_min: Optional[float], radius_max: Optional[float]) -> np.ndarray:
    if radius_min is None or radius_max is None:
        raise ValueError("Normalized radius actions require radius_min and radius_max")
    return float(radius_min) + values.astype(float) * (float(radius_max) - float(radius_min))


__all__ = ["render_radius_animation"]
