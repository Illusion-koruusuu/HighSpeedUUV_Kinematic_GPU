"""Generate batch previews of uniformly sampled obstacle scenes."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from kinematic_uuv.env import KinematicUUVEnv


ORANGE = "#F59E0B"
PALE_ORANGE = "#FDBA74"
NAVY = "#17365D"
BLUE = "#2563EB"


def sample_scene(seed, env_kwargs=None):
    env = KinematicUUVEnv(max_steps=2, **(env_kwargs or {}))
    try:
        env.reset(seed=int(seed))
        start = env.state["position"].copy()
        goal = env.goal.copy()
        obstacles = [
            (np.asarray(center, dtype=float).copy(), float(radius))
            for center, radius in env.obstacles
        ]
        bounds = (
            env.planning_bounds[0].copy(),
            env.planning_bounds[1].copy(),
        )
        reachable = bool(np.isfinite(env._dijkstra_cost(start)))
        occupancy_ratio = float(env._cost_occupancy.mean())
        clearances = []
        for center, radius in env.obstacles:
            clearances.append(
                env._segment_distance(center, start, goal) - radius
            )
    finally:
        env.close()
    return {
        "seed": int(seed),
        "start": start,
        "goal": goal,
        "obstacles": obstacles,
        "bounds": bounds,
        "reachable": reachable,
        "dijkstra_occupancy_ratio": occupancy_ratio,
        "direct_path_min_clearance_m": float(min(clearances)),
        "direct_path_blocked_count": int(sum(value <= 0.0 for value in clearances)),
        "direct_path_inflated_count": int(sum(value <= 15.0 for value in clearances)),
    }


def _draw_3d_scene(ax, scene):
    centers = np.asarray([center for center, _ in scene["obstacles"]])
    radii = np.asarray([radius for _, radius in scene["obstacles"]])
    ax.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2],
        s=np.clip(radii ** 2 * 0.8, 30, 500),
        c=ORANGE, alpha=0.55, edgecolor="#92400E", linewidth=0.4,
        label="obstacles",
    )
    ax.scatter(*scene["start"], marker="*", s=100, c="black", label="start")
    ax.scatter(*scene["goal"], marker="X", s=80, c="limegreen", label="goal")
    lower, upper = scene["bounds"]
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(upper - lower)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=25, azim=-62)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("3D distribution")


def _draw_projection(ax, scene, horizontal, vertical, equal=True):
    lower, upper = scene["bounds"]
    ax.plot(
        [scene["start"][horizontal], scene["goal"][horizontal]],
        [scene["start"][vertical], scene["goal"][vertical]],
        color=BLUE, linestyle="--", linewidth=1.0, alpha=0.65,
        label="direct line",
    )
    for center, radius in scene["obstacles"]:
        ax.add_patch(Circle(
            (center[horizontal], center[vertical]), radius,
            facecolor=ORANGE, edgecolor="#92400E", alpha=0.45, linewidth=0.5,
        ))
        ax.add_patch(Circle(
            (center[horizontal], center[vertical]), radius + 15.0,
            fill=False, edgecolor=PALE_ORANGE, linestyle="--",
            alpha=0.60, linewidth=0.45,
        ))
    ax.scatter(
        scene["start"][horizontal], scene["start"][vertical],
        marker="*", s=90, c="black", zorder=5, label="start",
    )
    ax.scatter(
        scene["goal"][horizontal], scene["goal"][vertical],
        marker="X", s=70, c="limegreen", edgecolor="darkgreen",
        zorder=5, label="goal",
    )
    ax.add_patch(Rectangle(
        (lower[horizontal], lower[vertical]),
        upper[horizontal] - lower[horizontal],
        upper[vertical] - lower[vertical],
        fill=False, edgecolor=NAVY, linewidth=0.7, linestyle=":",
    ))
    ax.set_xlim(lower[horizontal], upper[horizontal])
    ax.set_ylim(lower[vertical], upper[vertical])
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    ax.set_xlabel(("x", "y", "z")[horizontal] + " (m)")
    ax.set_ylabel(("x", "y", "z")[vertical] + " (m)")
    ax.legend(loc="upper left", fontsize=6)


def plot_scene(scene, output):
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.25, 1.0))
    ax_xy = fig.add_subplot(grid[0, :])
    ax3d = fig.add_subplot(grid[1, 0], projection="3d")
    ax_xz = fig.add_subplot(grid[1, 1])
    ax_yz = fig.add_subplot(grid[1, 2])
    _draw_3d_scene(ax3d, scene)
    _draw_projection(ax_xy, scene, 0, 1)
    _draw_projection(ax_xz, scene, 0, 2)
    _draw_projection(ax_yz, scene, 1, 2)
    ax_xy.set_title("Top view: XY")
    ax_xz.set_title("Side view: XZ")
    ax_yz.set_title("Side view: YZ")
    radii = np.asarray([radius for _, radius in scene["obstacles"]])
    fig.suptitle(
        f"Scene seed={scene['seed']} | count={len(radii)} | "
        f"radius mean={radii.mean():.1f} m, max={radii.max():.1f} m | "
        f"reachable={scene['reachable']}",
        color=NAVY, fontsize=16,
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_overview(scenes, output):
    columns = min(4, len(scenes))
    rows = math.ceil(len(scenes) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.0 * columns, 4.2 * rows),
        squeeze=False, constrained_layout=True,
    )
    for ax, scene in zip(axes.flat, scenes):
        _draw_projection(ax, scene, 0, 1)
        radii = np.asarray([radius for _, radius in scene["obstacles"]])
        ax.set_title(
            f"seed={scene['seed']}  n={len(radii)}  "
            f"r={radii.mean():.1f} m  "
            f"block={scene['direct_path_blocked_count']}",
            fontsize=9,
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    for ax in axes.flat[len(scenes):]:
        ax.axis("off")
    fig.suptitle(
        "Uniform dense obstacle scenes - top view",
        fontsize=18, color=NAVY,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("output_model/scene_preview_uniform"))
    parser.add_argument("--num-scenes", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--min-obstacles", type=int, default=300)
    parser.add_argument("--max-obstacles", type=int, default=450)
    parser.add_argument("--obstacle-radius-min", type=float, default=4.0)
    parser.add_argument("--obstacle-radius-max", type=float, default=22.0)
    args = parser.parse_args()
    if args.num_scenes <= 0:
        parser.error("num-scenes must be positive")
    if (args.min_obstacles < 0
            or args.max_obstacles < args.min_obstacles
            or args.obstacle_radius_min <= 0
            or args.obstacle_radius_max < args.obstacle_radius_min):
        parser.error("invalid obstacle count or radius range")
    env_kwargs = {
        "obstacle_count_range": (args.min_obstacles, args.max_obstacles),
        "obstacle_radius_range": (args.obstacle_radius_min,
                                  args.obstacle_radius_max),
        "obstacle_distribution": "uniform",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    scenes = [
        sample_scene(args.seed_start + index, env_kwargs)
        for index in range(args.num_scenes)
    ]
    summary = []
    for scene in scenes:
        radii = np.asarray([radius for _, radius in scene["obstacles"]])
        image = args.output / f"scene_seed{scene['seed']}.png"
        plot_scene(scene, image)
        summary.append({
            "seed": scene["seed"],
            "obstacle_count": int(len(radii)),
            "radius_min_m": float(radii.min()),
            "radius_mean_m": float(radii.mean()),
            "radius_max_m": float(radii.max()),
            "large_radius_count": int((radii >= 14.0).sum()),
            "reachable": scene["reachable"],
            "dijkstra_occupancy_ratio": scene["dijkstra_occupancy_ratio"],
            "direct_path_min_clearance_m": scene["direct_path_min_clearance_m"],
            "direct_path_blocked_count": scene["direct_path_blocked_count"],
            "direct_path_inflated_count": scene["direct_path_inflated_count"],
            "image": str(image),
        })
        print(f"Saved {image}", flush=True)

    overview = args.output / "overview_xy.png"
    plot_overview(scenes, overview)
    (args.output / "scene_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {overview}")


if __name__ == "__main__":
    main()
