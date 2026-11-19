"""Play a recorded UUV trajectory in real time or export it as a video."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

from kinematic_uuv.env import KNOT_MPS


def _draw_sphere(ax, center, radius, color, label, alpha=0.2):
    u = np.linspace(0.0, 2.0 * np.pi, 18)
    v = np.linspace(0.0, np.pi, 12)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)
    ax.plot([], [], color=color, label=label)


def _make_animation(data, args):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    position = np.asarray(data["position"], dtype=float)
    speed = np.asarray(data["speed"], dtype=float)
    rotation = np.asarray(data["roll_pitch_yaw"], dtype=float)
    obstacles = np.asarray(data["obstacles"], dtype=float)
    goal = np.asarray(data["goal"], dtype=float)
    goal_radius = float(data.get("goal_radius_m", 5.0))
    dt = float(data.get("dt", args.dt))
    total_frames = len(position)
    end_step = min(total_frames - 1, args.end_step if args.end_step else total_frames - 1)
    start_step = max(0, min(args.start_step, end_step))
    frame_indices = np.arange(start_step, end_step + 1, args.stride)
    time = np.arange(total_frames) * dt
    trail_frames = max(1, int(args.trail_seconds / dt))

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(2.0, 1.0))
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_speed = fig.add_subplot(grid[0, 1])

    for index, obstacle in enumerate(obstacles):
        _draw_sphere(
            ax3d, obstacle[:3], float(obstacle[3]), "orange",
            "obstacle" if index == 0 else "_nolegend_",
        )
    _draw_sphere(ax3d, goal, goal_radius, "limegreen", "goal", alpha=0.28)
    ax3d.scatter(*position[start_step], color="black", s=45, label="start")
    ax3d.scatter(*position[end_step], color="red", s=45, label="end")
    trail, = ax3d.plot([], [], [], color="royalblue", linewidth=2.0,
                        label="traveled path")
    marker, = ax3d.plot([], [], [], "o", color="red", markersize=8,
                         label="AUV")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.invert_zaxis()
    ax3d.view_init(elev=args.elevation, azim=args.azimuth)

    plot_bounds = np.vstack((
        position,
        goal[None, :] - goal_radius,
        goal[None, :] + goal_radius,
        obstacles[:, :3] - obstacles[:, 3:4],
        obstacles[:, :3] + obstacles[:, 3:4],
    ))
    spans = np.maximum(plot_bounds.max(axis=0) - plot_bounds.min(axis=0), 1e-6)
    ax3d.set_box_aspect(spans)
    ax3d.legend(loc="upper left", fontsize=8)

    ax_speed.plot(time, speed / KNOT_MPS, color="royalblue",
                  label="actual speed")
    ax_speed.axhline(
        float(data["target_speed_knots"]), color="black",
        linestyle="--", label="target speed",
    )
    speed_cursor = ax_speed.axvline(time[start_step], color="red")
    ax_speed.set_xlim(time[start_step], time[end_step])
    ax_speed.set_ylim(0.0, max(speed.max() / KNOT_MPS, 1.0) * 1.15)
    ax_speed.set_xlabel("time (s)")
    ax_speed.set_ylabel("speed (knots)")
    ax_speed.grid(True)
    ax_speed.legend()

    status = fig.text(
        0.01, 0.01, "", fontfamily="monospace", fontsize=10,
    )
    orientation_arrow = [None]

    def update(frame):
        index = int(frame_indices[frame])
        trail_start = max(start_step, index - trail_frames)
        trail.set_data(
            position[trail_start:index + 1, 0],
            position[trail_start:index + 1, 1],
        )
        trail.set_3d_properties(position[trail_start:index + 1, 2])
        marker.set_data([position[index, 0]], [position[index, 1]])
        marker.set_3d_properties([position[index, 2]])

        if orientation_arrow[0] is not None:
            orientation_arrow[0].remove()
        pitch = float(rotation[index, 1])
        yaw = float(rotation[index, 2])
        direction = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            -np.sin(pitch),
        ])
        orientation_arrow[0] = ax3d.quiver(
            *position[index], *direction, length=10.0, color="crimson",
            normalize=True,
        )
        speed_cursor.set_xdata([time[index], time[index]])
        status.set_text(
            f"t={time[index]:6.2f}s  step={index:4d}  "
            f"speed={speed[index] / KNOT_MPS:5.1f}kn  "
            f"pitch={np.rad2deg(pitch):6.2f}deg  "
            f"yaw={np.rad2deg(yaw):7.2f}deg"
        )
        ax3d.set_title(
            f"{args.trajectory.stem}  t={time[index]:.1f}s"
        )
        return trail, marker, speed_cursor, status

    interval = max(1, int(round(1000.0 * dt / args.speed)))
    animation = FuncAnimation(
        fig, update, frames=len(frame_indices), interval=interval,
        repeat=args.repeat, blit=False,
    )
    return fig, animation, interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path,
                        help=".npz trajectory generated by training.evaluate")
    parser.add_argument("--save", type=Path,
                        help="optional .gif or .mp4 output instead of a window")
    parser.add_argument("--stride", type=int, default=1,
                        help="play every Nth recorded frame")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed multiplier")
    parser.add_argument("--trail-seconds", type=float, default=8.0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--dt", type=float, default=0.1,
                        help="fallback for old trajectories without dt")
    parser.add_argument("--backend",
                        help="interactive backend, for example TkAgg")
    parser.add_argument("--elevation", type=float, default=28.0)
    parser.add_argument("--azimuth", type=float, default=-65.0)
    parser.add_argument("--repeat", action=argparse.BooleanOptionalAction,
                        default=False)
    args = parser.parse_args()
    if not args.trajectory.is_file():
        parser.error(f"trajectory does not exist: {args.trajectory}")
    if args.stride <= 0 or args.speed <= 0.0 or args.trail_seconds <= 0.0:
        parser.error("stride, speed and trail-seconds must be positive")

    if args.save:
        matplotlib.use("Agg", force=True)
    elif args.backend:
        matplotlib.use(args.backend, force=True)
    import matplotlib.pyplot as plt

    with np.load(args.trajectory) as archive:
        data = {key: archive[key] for key in archive.files}
    fig, animation, interval = _make_animation(data, args)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        suffix = args.save.suffix.lower()
        fps = max(1.0, 1000.0 / interval)
        if suffix == ".gif":
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
        elif suffix == ".mp4":
            writer = "ffmpeg"
        else:
            parser.error("--save must use .gif or .mp4")
        print(
            f"Exporting {args.save} at {fps:.1f} FPS...",
            flush=True,
        )
        save_options = {"dpi": 130}
        if suffix == ".mp4":
            save_options["bitrate"] = 2400
        animation.save(args.save, writer=writer, **save_options)
        plt.close(fig)
        print(f"Saved {args.save}")
        return

    if matplotlib.get_backend().lower() == "agg":
        parser.error(
            "no interactive Matplotlib backend is available; "
            "run with --backend TkAgg or export with --save trajectory.mp4"
        )
    fig.suptitle("Close the window to stop playback")
    plt.show()


if __name__ == "__main__":
    main()
