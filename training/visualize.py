"""Create offline trajectory diagnostics from evaluation .npz files."""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kinematic_uuv.env import KNOT_MPS


def _trajectory_files(path, speeds, seeds, max_trajectories,
                      max_per_speed=None):
    if path.is_dir():
        search_dir = path / "trajectories" if (path / "trajectories").is_dir() else path
        files = sorted(search_dir.glob("*.npz"))
    else:
        files = [path]
    if speeds:
        filtered = []
        for file in files:
            match = re.match(r"speed([0-9.]+)_", file.stem)
            if match and float(match.group(1)) in speeds:
                filtered.append(file)
        files = filtered
    if seeds:
        filtered = []
        for file in files:
            match = re.search(r"_seed([0-9]+)\.npz$", file.name)
            if match and int(match.group(1)) in seeds:
                filtered.append(file)
        files = filtered
    if max_per_speed is not None:
        selected = []
        counts = {}
        for file in files:
            match = re.match(r"speed([0-9.]+)_", file.stem)
            if not match:
                continue
            speed = float(match.group(1))
            if counts.get(speed, 0) >= max_per_speed:
                continue
            counts[speed] = counts.get(speed, 0) + 1
            selected.append(file)
        files = selected
    return files[:max_trajectories]


def _load(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _clearance(position, obstacles):
    if len(obstacles) == 0:
        return np.full(len(position), np.inf)
    distances = np.linalg.norm(
        position[:, None, :] - obstacles[None, :, :3], axis=2
    )
    return np.min(distances - obstacles[None, :, 3], axis=1)


def _summary(data, dt):
    position = np.asarray(data["position"], dtype=float)
    speed = np.asarray(data["speed"], dtype=float)
    rotation = np.asarray(data["roll_pitch_yaw"], dtype=float)
    yaw_rate = np.asarray(data["yaw_rate"], dtype=float)
    goal = np.asarray(data["goal"], dtype=float)
    goal_radius = float(data.get("goal_radius_m", 5.0))
    obstacles = np.asarray(data["obstacles"], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(position, axis=0), axis=1)
    return {
        "success": bool(data["success"]),
        "target_speed_knots": float(data["target_speed_knots"]),
        "seed": int(data["seed"]),
        "steps": int(len(speed) - 1),
        "duration_s": float((len(speed) - 1) * dt),
        "path_length_m": float(segment_lengths.sum()),
        "final_distance_to_goal_m": float(np.linalg.norm(position[-1] - goal)),
        "goal_radius_m": goal_radius,
        "min_obstacle_clearance_m": float(_clearance(position, obstacles).min()),
        "max_abs_pitch_deg": float(np.rad2deg(np.abs(rotation[:, 1]).max())),
        "max_abs_yaw_rate_deg_s": float(np.rad2deg(np.abs(yaw_rate).max())),
        "max_lat_accel_mps2": float(np.max(np.abs(speed * yaw_rate))),
        "max_speed_knots": float(speed.max() / KNOT_MPS),
    }


def _draw_sphere(ax, center, radius, color, label, alpha=0.18):
    u = np.linspace(0.0, 2.0 * np.pi, 16)
    v = np.linspace(0.0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)
    ax.plot([], [], color=color, label=label)


def plot_trajectory(data, path, dt):
    position = np.asarray(data["position"], dtype=float)
    speed = np.asarray(data["speed"], dtype=float)
    rotation = np.asarray(data["roll_pitch_yaw"], dtype=float)
    yaw_rate = np.asarray(data["yaw_rate"], dtype=float)
    obstacles = np.asarray(data["obstacles"], dtype=float)
    goal = np.asarray(data["goal"], dtype=float)
    goal_radius = float(data.get("goal_radius_m", 5.0))
    time = np.arange(len(speed)) * dt
    distance = np.linalg.norm(position - goal[None, :], axis=1)
    clearance = _clearance(position, obstacles)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_speed = fig.add_subplot(grid[0, 1])
    ax_attitude = fig.add_subplot(grid[1, 1])

    points = ax3d.scatter(
        position[:, 0], position[:, 1], position[:, 2],
        c=speed / KNOT_MPS, cmap="viridis", s=8,
    )
    ax3d.plot(position[:, 0], position[:, 1], position[:, 2],
              color="royalblue", linewidth=1.0, alpha=0.7)
    for index, obstacle in enumerate(obstacles):
        _draw_sphere(
            ax3d, obstacle[:3], float(obstacle[3]), "orange",
            "obstacle" if index == 0 else "_nolegend_",
        )
    _draw_sphere(ax3d, goal, goal_radius, "limegreen", "goal", alpha=0.25)
    ax3d.scatter(*position[0], color="black", s=55, label="start")
    ax3d.scatter(*position[-1], color="red", s=55, label="end")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.invert_zaxis()
    ax3d.view_init(elev=28, azim=-65)
    plot_bounds = np.vstack((
        position,
        goal[None, :] - goal_radius,
        goal[None, :] + goal_radius,
        obstacles[:, :3] - obstacles[:, 3:4],
        obstacles[:, :3] + obstacles[:, 3:4],
    ))
    spans = np.maximum(plot_bounds.max(axis=0) - plot_bounds.min(axis=0), 1e-6)
    ax3d.set_box_aspect(spans)
    ax3d.set_title(
        f"{path.stem}: {'SUCCESS' if bool(data['success']) else 'FAILED'}"
    )
    ax3d.legend(loc="upper left", fontsize=8)
    fig.colorbar(
        points, ax=ax3d, orientation="horizontal", shrink=0.72,
        pad=0.08, label="speed (knots)",
    )

    ax_speed.plot(time, speed / KNOT_MPS, label="actual speed")
    ax_speed.axhline(
        float(data["target_speed_knots"]), color="black",
        linestyle="--", label="target cruise speed",
    )
    ax_speed.set_xlabel("time (s)")
    ax_speed.set_ylabel("speed (knots)")
    ax_speed.grid(True)
    ax_speed.legend()
    ax_speed.set_title("Speed tracking")

    ax_attitude.plot(time, np.rad2deg(rotation[:, 1]), label="pitch")
    ax_attitude.plot(time, np.rad2deg(rotation[:, 2]), label="yaw")
    ax_attitude.plot(
        time, np.rad2deg(yaw_rate),
        linestyle=":", label="yaw rate",
    )
    ax_attitude.set_xlabel("time (s)")
    ax_attitude.set_ylabel("angle (deg or deg/s)")
    ax_attitude.grid(True)
    ax_attitude.legend()
    ax_attitude.set_title("Attitude")
    ax_distance = ax_attitude.twinx()
    ax_distance.plot(time, distance, color="crimson", alpha=0.55,
                     label="distance to goal")
    ax_distance.plot(time, clearance, color="darkorange", alpha=0.55,
                     label="obstacle clearance")
    ax_distance.set_ylabel("distance / clearance (m)")
    lines = ax_attitude.get_lines() + ax_distance.get_lines()
    ax_attitude.legend(lines, [line.get_label() for line in lines], fontsize=8)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory", type=Path,
        help=".npz file, trajectory directory, or evaluation output directory",
    )
    parser.add_argument("--output", type=Path,
                        default=Path("output_model/kinematic_visualization"))
    parser.add_argument("--max-trajectories", type=int, default=20)
    parser.add_argument(
        "--max-per-speed", type=int,
        help="optional limit on selected trajectories for each speed",
    )
    parser.add_argument("--speeds-knots", type=float, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()
    if not args.trajectory.exists():
        parser.error(f"trajectory path does not exist: {args.trajectory}")
    if (args.max_trajectories <= 0 or args.dt <= 0.0
            or (args.max_per_speed is not None and args.max_per_speed <= 0)):
        parser.error("max-trajectories and dt must be positive")

    files = _trajectory_files(
        args.trajectory, args.speeds_knots, args.seeds,
        args.max_trajectories, args.max_per_speed,
    )
    if not files:
        parser.error("no matching .npz trajectory files found")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for file in files:
        data = _load(file)
        trajectory_dt = float(data.get("dt", args.dt))
        output = args.output / f"{file.stem}_diagnostics.png"
        plot_trajectory(data, output, trajectory_dt)
        summary = _summary(data, trajectory_dt)
        summary["source"] = str(file)
        summary["image"] = str(output)
        summaries.append(summary)
        print(
            f"{file.name}: success={summary['success']} "
            f"clearance={summary['min_obstacle_clearance_m']:.2f}m "
            f"final={summary['final_distance_to_goal_m']:.2f}m -> {output}",
            flush=True,
        )

    summary_path = args.output / "visualization_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
