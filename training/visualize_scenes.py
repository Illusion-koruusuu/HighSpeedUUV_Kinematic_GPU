"""Generate batch previews of procedural seabed terrain scenes."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kinematic_uuv.env import KinematicUUVEnv


NAVY = "#17365D"
BLUE = "#2563EB"
GREEN = "#16A34A"


def sample_scene(seed, env_kwargs=None):
    env = KinematicUUVEnv(max_steps=2, **(env_kwargs or {}))
    try:
        env.reset(seed=int(seed))
        start = env.state["position"].copy()
        goal = env.goal.copy()
        terrain = env.terrain
        bounds = (
            env.planning_bounds[0].copy(),
            env.planning_bounds[1].copy(),
        )
        path = env._current_path().copy()
        fractions = np.linspace(0.0, 1.0, 121)
        direct_points = (
            start[None, :] + fractions[:, None] * (goal - start)[None, :]
        )
        direct_surface = terrain.height(
            direct_points[:, 0], direct_points[:, 1]
        )
        direct_clearance = direct_points[:, 2] - direct_surface
        return {
            "seed": int(seed),
            "start": start,
            "goal": goal,
            "path": path,
            "bounds": bounds,
            "terrain_x": terrain.x_axis.copy(),
            "terrain_y": terrain.y_axis.copy(),
            "terrain_height": terrain.height_grid.copy(),
            "reachable": bool(np.isfinite(env._dijkstra_cost(start))),
            "dijkstra_occupancy_ratio": float(env._cost_occupancy.mean()),
            "direct_path_min_clearance_m": float(
                direct_clearance.min()
            ),
            "direct_path_blocked": bool((direct_clearance <= 0.0).any()),
            "start_clearance_m": terrain.clearance(start),
            "goal_clearance_m": terrain.clearance(goal),
            "direct_points": direct_points,
            "direct_surface": direct_surface,
            "direct_clearance": direct_clearance,
        }
    finally:
        env.close()


def _draw_3d_scene(ax, scene):
    x = scene["terrain_x"]
    y = scene["terrain_y"]
    height = scene["terrain_height"]
    grid_x, grid_y = np.meshgrid(x, y)
    surface = ax.plot_surface(
        grid_x, grid_y, height,
        cmap="terrain", linewidth=0, antialiased=True,
        alpha=0.92, vmin=0.0,
        vmax=max(float(height.max()), 1.0),
    )
    ax.plot(
        scene["start"][0], scene["start"][1], scene["start"][2],
        marker="*", markersize=12, color="black", linestyle="None",
        label="start",
    )
    ax.plot(
        scene["goal"][0], scene["goal"][1], scene["goal"][2],
        marker="X", markersize=10, color=GREEN, linestyle="None",
        label="goal",
    )
    ax.plot(
        [scene["start"][0], scene["goal"][0]],
        [scene["start"][1], scene["goal"][1]],
        [scene["start"][2], scene["goal"][2]],
        color=BLUE, linestyle="--", linewidth=1.2, alpha=0.75,
        label="direct line",
    )
    if len(scene["path"]) >= 2:
        ax.plot(
            scene["path"][:, 0], scene["path"][:, 1], scene["path"][:, 2],
            color="#7C3AED", linewidth=2.0, label="teacher Dijkstra path",
        )
    lower, upper = scene["bounds"]
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(upper - lower)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=28, azim=-62)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("3D seabed terrain")
    return surface


def _draw_top_view(ax, scene):
    x = scene["terrain_x"]
    y = scene["terrain_y"]
    height = scene["terrain_height"]
    image = ax.imshow(
        height, origin="lower", cmap="terrain", aspect="equal",
        extent=(x[0], x[-1], y[0], y[-1]),
        vmin=0.0, vmax=max(float(height.max()), 1.0),
    )
    contour = ax.contour(
        x, y, height, levels=8, colors="black",
        linewidths=0.45, alpha=0.45,
    )
    ax.clabel(contour, inline=True, fontsize=6, fmt="%.0f")
    ax.plot(
        [scene["start"][0], scene["goal"][0]],
        [scene["start"][1], scene["goal"][1]],
        color=BLUE, linestyle="--", linewidth=1.2, label="direct line",
    )
    if len(scene["path"]) >= 2:
        ax.plot(
            scene["path"][:, 0], scene["path"][:, 1],
            color="#7C3AED", linewidth=2.0, label="Dijkstra path",
        )
    ax.scatter(*scene["start"][:2], marker="*", s=100, c="black", label="start")
    ax.scatter(
        *scene["goal"][:2], marker="X", s=85, c=GREEN,
        edgecolor="darkgreen", label="goal",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Top view and contour lines")
    ax.legend(loc="upper left", fontsize=7)
    return image


def _draw_direct_profile(ax, scene):
    points = scene["direct_points"]
    horizontal = np.hypot(
        points[:, 0] - points[0, 0],
        points[:, 1] - points[0, 1],
    )
    ax.fill_between(
        horizontal, 0.0, scene["direct_surface"],
        color="#A16207", alpha=0.55, label="seabed cross-section",
    )
    ax.plot(
        horizontal, points[:, 2],
        color=BLUE, linewidth=1.6, label="straight vehicle path",
    )
    ax.plot(
        horizontal, scene["direct_surface"],
        color="#78350F", linewidth=1.0,
    )
    ax.set_xlabel("distance along direct line (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Direct-route clearance")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)


def plot_scene(scene, output):
    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.2, 1.0))
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_top = fig.add_subplot(grid[0, 1])
    ax_profile = fig.add_subplot(grid[1, 1])
    surface = _draw_3d_scene(ax3d, scene)
    image = _draw_top_view(ax_top, scene)
    _draw_direct_profile(ax_profile, scene)
    fig.colorbar(
        surface, ax=ax3d, shrink=0.55, pad=0.09,
        label="seabed height (m)",
    )
    fig.colorbar(
        image, ax=ax_top, shrink=0.82, pad=0.02,
        label="height (m)",
    )
    fig.suptitle(
        f"Terrain seed={scene['seed']} | "
        f"height {scene['terrain_height'].min():.1f}–"
        f"{scene['terrain_height'].max():.1f} m | "
        f"direct blocked={scene['direct_path_blocked']} | "
        f"reachable={scene['reachable']}",
        color=NAVY, fontsize=15,
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_overview(scenes, output):
    columns = min(3, len(scenes))
    rows = math.ceil(len(scenes) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.2 * columns, 4.0 * rows),
        squeeze=False, constrained_layout=True,
    )
    for ax, scene in zip(axes.flat, scenes):
        _draw_top_view(ax, scene)
        ax.set_title(
            f"seed={scene['seed']} | "
            f"max={scene['terrain_height'].max():.0f} m | "
            f"blocked={scene['direct_path_blocked']}",
            fontsize=9,
        )
    for ax in axes.flat[len(scenes):]:
        ax.axis("off")
    fig.suptitle("Procedural seabed terrain scenes", fontsize=18, color=NAVY)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("output_model/terrain_preview"))
    parser.add_argument("--num-scenes", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=7000)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--terrain-resolution", type=float, default=20.0)
    parser.add_argument("--min-terrain-peaks", type=int, default=18)
    parser.add_argument("--max-terrain-peaks", type=int, default=30)
    parser.add_argument("--terrain-base-height-min", type=float, default=1.0)
    parser.add_argument("--terrain-base-height-max", type=float, default=7.0)
    parser.add_argument("--terrain-max-height", type=float, default=62.0)
    parser.add_argument("--terrain-safety-margin", type=float, default=8.0)
    parser.add_argument("--terrain-goal-clearance", type=float, default=12.0)
    parser.add_argument("--terrain-cruise-height", type=float, default=30.0)
    args = parser.parse_args()
    if args.num_scenes <= 0:
        parser.error("num-scenes must be positive")
    if args.seeds and len(args.seeds) != args.num_scenes:
        parser.error("--seeds must contain exactly num-scenes values")
    if args.terrain_resolution <= 0.0:
        parser.error("terrain-resolution must be positive")
    if (args.min_terrain_peaks < 0
            or args.max_terrain_peaks < args.min_terrain_peaks):
        parser.error("invalid terrain peak range")
    if args.terrain_max_height <= args.terrain_base_height_max:
        parser.error("terrain-max-height must exceed base height")
    if args.terrain_cruise_height <= 0.0:
        parser.error("terrain-cruise-height must be positive")

    env_kwargs = {
        "scene_type": "terrain",
        "terrain_resolution": args.terrain_resolution,
        "terrain_peak_count_range": (
            args.min_terrain_peaks, args.max_terrain_peaks
        ),
        "terrain_base_height_range": (
            args.terrain_base_height_min, args.terrain_base_height_max
        ),
        "terrain_max_height": args.terrain_max_height,
        "terrain_safety_margin": args.terrain_safety_margin,
        "terrain_goal_clearance": args.terrain_goal_clearance,
        "terrain_cruise_height": args.terrain_cruise_height,
        "sensor_backend": "cpu",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = (
        list(args.seeds)
        if args.seeds is not None
        else [args.seed_start + index for index in range(args.num_scenes)]
    )
    scenes = [sample_scene(seed, env_kwargs) for seed in seeds]
    summary = []
    for scene in scenes:
        image = args.output / f"terrain_seed{scene['seed']}.png"
        plot_scene(scene, image)
        summary.append({
            "seed": scene["seed"],
            "terrain_height_min_m": float(scene["terrain_height"].min()),
            "terrain_height_mean_m": float(scene["terrain_height"].mean()),
            "terrain_height_max_m": float(scene["terrain_height"].max()),
            "start_clearance_m": scene["start_clearance_m"],
            "goal_clearance_m": scene["goal_clearance_m"],
            "direct_path_blocked": scene["direct_path_blocked"],
            "direct_path_min_clearance_m":
                scene["direct_path_min_clearance_m"],
            "reachable": scene["reachable"],
            "dijkstra_occupancy_ratio":
                scene["dijkstra_occupancy_ratio"],
            "image": str(image),
        })
        print(f"Saved {image}", flush=True)

    overview = args.output / "overview_terrain.png"
    plot_overview(scenes, overview)
    (args.output / "terrain_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {overview}")


if __name__ == "__main__":
    main()
