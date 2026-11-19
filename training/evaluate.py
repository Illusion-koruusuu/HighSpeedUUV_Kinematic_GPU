"""Evaluate a kinematic UUV policy across low and high target speeds."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from kinematic_uuv.env import KNOT_MPS, KinematicUUVEnv
from kinematic_uuv.policy import BoundedActorCriticPolicy, SonarCNNFeaturesExtractor
from kinematic_uuv.runtime import configure_torch, resolve_device


def _load_config(model_path):
    path = model_path.parent / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"run.json not found next to model: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _make_env(config, episode_steps):
    bounds = config.get("planning_bounds_m")
    kwargs = dict(
        dt=config.get("dt", 0.1),
        max_steps=episode_steps,
        speed_min=config.get("speed_min_mps", 1.0),
        speed_max=config.get("speed_max_mps", 50.0 * KNOT_MPS),
        cruise_speed_range=tuple(config.get("cruise_speed_range_mps",
                                            [5.0 * KNOT_MPS, 50.0 * KNOT_MPS])),
        max_accel=config.get("max_accel_mps2", 2.5),
        max_decel=config.get("max_decel_mps2", 3.0),
        max_yaw_rate=config.get("max_yaw_rate_rad_s", 0.8),
        max_yaw_accel=config.get("max_yaw_accel_rad_s2", 1.0),
        max_lat_accel=config.get("max_lat_accel_mps2", 4.0),
        min_turn_radius=config.get("min_turn_radius_m", 25.0),
        max_pitch_rate=config.get("max_pitch_rate_rad_s", 0.20),
        max_pitch=config.get("max_pitch_rad", 0.35),
        goal_radius=config.get("goal_radius_m", 5.0),
        obstacle_count_range=tuple(config.get("obstacle_count_range", [2, 4])),
        obstacle_radius_range=tuple(
            config.get("obstacle_radius_range_m", [4.0, 8.0])
        ),
        obstacle_distribution=config.get(
            "obstacle_distribution", "uniform"
        ),
        sensor_backend=config.get("sensor_backend", "cpu"),
        sensor_device=config.get("sensor_device", "cuda"),
        goal_distance_range=tuple(config.get("goal_distance_range_m", [120, 450])),
        dijkstra_resolution=config.get("dijkstra_resolution_m", 8.0),
        path_refresh_steps=config.get("path_refresh_steps", 5),
    )
    if bounds is not None:
        kwargs["planning_bounds"] = (bounds[0], bounds[1])
    return KinematicUUVEnv(**kwargs)


def _save_trajectory(path, record):
    np.savez_compressed(
        path,
        position=np.asarray(record["position"]),
        roll_pitch_yaw=np.asarray(record["roll_pitch_yaw"]),
        speed=np.asarray(record["speed"]),
        actions=np.asarray(record["actions"]),
        yaw_rate=np.asarray(record["yaw_rate"]),
        goal=np.asarray(record["goal"]),
        goal_radius_m=np.asarray(record["goal_radius_m"]),
        obstacles=np.asarray(record["obstacles"]),
        cruise_speed_mps=np.asarray(record["cruise_speed_mps"]),
        success=np.asarray(record["success"]),
        seed=np.asarray(record["seed"]),
        target_speed_knots=np.asarray(record["target_speed_knots"]),
        dt=np.asarray(record.get("dt", 0.1)),
    )


def run_episode(env, model, seed, speed_knots):
    env.cruise_speed_override = speed_knots * KNOT_MPS
    observation, _ = env.reset(seed=seed)
    position = [env.state["position"].copy()]
    rotation = [[0.0, env.state["pitch"], env.state["yaw"]]]
    speed = [env.state["speed"]]
    actions = []
    yaw_rate = [env.state["yaw_rate"]]
    max_clearance = float("inf")
    total_reward = 0.0
    steps = 0
    info = {}
    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        actions.append(np.asarray(action, dtype=float))
        position.append(env.state["position"].copy())
        rotation.append([0.0, env.state["pitch"], env.state["yaw"]])
        speed.append(env.state["speed"])
        yaw_rate.append(env.state["yaw_rate"])
        max_clearance = min(max_clearance, float(info["clearance_m"]))
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    result = {
        "target_speed_knots": float(speed_knots),
        "seed": int(seed),
        "success": int(bool(info.get("reached", False))),
        "collision": int(bool(info.get("collision", False))),
        "out_of_bounds": int(bool(info.get("out_of_bounds", False))),
        "timeout": int(truncated and not terminated),
        "steps": steps,
        "time_s": steps * env.dt,
        "reward": float(total_reward),
        "final_distance_m": float(info.get("distance_to_goal", np.nan)),
        "mean_speed_mps": float(np.mean(speed)),
        "max_speed_mps": float(np.max(speed)),
        "min_clearance_m": max_clearance,
        "max_abs_yaw_rate_rad_s": float(np.max(np.abs(yaw_rate))),
        "max_lat_accel_mps2": float(np.max(
            np.asarray(speed) * np.abs(np.asarray(yaw_rate))
        )),
        "max_abs_pitch_rad": float(np.max(np.abs(np.asarray(rotation)[:, 1]))),
        "path_length_m": float(np.linalg.norm(
            np.diff(np.asarray(position), axis=0), axis=1
        ).sum()),
    }
    trajectory = {
        "position": position,
        "roll_pitch_yaw": rotation,
        "speed": speed,
        "actions": actions,
        "yaw_rate": yaw_rate,
        "goal": env.goal.copy(),
        "goal_radius_m": env.goal_radius,
        "obstacles": np.asarray([
            [*center, radius] for center, radius in env.obstacles
        ]),
        "cruise_speed_mps": env.cruise_speed,
        "success": bool(info.get("reached", False)),
        "seed": seed,
        "target_speed_knots": speed_knots,
        "dt": env.dt,
    }
    return result, trajectory


def write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    speeds = sorted(set(row["target_speed_knots"] for row in rows))
    summary = []
    for speed in speeds:
        subset = [row for row in rows if row["target_speed_knots"] == speed]
        summary.append({
            "target_speed_knots": speed,
            "episodes": len(subset),
            "success_rate": float(np.mean([row["success"] for row in subset])),
            "collision_rate": float(np.mean([row["collision"] for row in subset])),
            "out_of_bounds_rate": float(np.mean([
                row["out_of_bounds"] for row in subset
            ])),
            "timeout_rate": float(np.mean([row["timeout"] for row in subset])),
            "mean_time_s": float(np.mean([row["time_s"] for row in subset])),
            "mean_speed_mps": float(np.mean([
                row["mean_speed_mps"] for row in subset
            ])),
            "max_lat_accel_mps2": float(np.max([
                row["max_lat_accel_mps2"] for row in subset
            ])),
        })
    return summary


def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(summary[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def plot_summary(summary, output):
    speeds = [row["target_speed_knots"] for row in summary]
    success = [100.0 * row["success_rate"] for row in summary]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(speeds, success, width=max(1.0, min(np.diff(speeds), default=5.0) * 0.55))
    ax.set_xlabel("Target speed (knots)")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y")
    ax.set_title("Kinematic policy success by speed")
    fig.tight_layout()
    path = output / "evaluation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("output_model/kinematic_eval"),
    )
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--episodes-per-speed", type=int, default=50)
    parser.add_argument("--speeds-knots", type=float, nargs="+",
                        default=[10.0, 20.0, 30.0, 40.0, 50.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-trajectories", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device: cuda, cuda:0, cpu, or auto")
    args = parser.parse_args()
    if args.episodes_per_speed <= 0 or args.episode_steps <= 0:
        parser.error("episode counts must be positive")

    config = _load_config(args.model)
    if config.get("observation_type") != "kinematic_sonar_path_v1":
        parser.error("model is not a kinematic_sonar_path_v1 policy")
    device = resolve_device(args.device)
    configure_torch(device)
    env = _make_env(config, args.episode_steps)
    model = PPO.load(str(args.model), device=str(device))
    if model.observation_space != env.observation_space:
        raise ValueError("model observation space differs from environment")
    args.output.mkdir(parents=True, exist_ok=True)
    csv_dir = args.output / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.output / "trajectories"
    if args.save_trajectories:
        trajectory_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    try:
        for speed in args.speeds_knots:
            for episode in range(args.episodes_per_speed):
                seed = args.seed + episode
                result, trajectory = run_episode(env, model, seed, speed)
                rows.append(result)
                if args.save_trajectories:
                    _save_trajectory(
                        trajectory_dir / f"speed{int(speed)}_seed{seed}.npz",
                        trajectory,
                    )
                print(
                    f"speed={speed:g} kn seed={seed} success={result['success']} "
                    f"collision={result['collision']} oob={result['out_of_bounds']} "
                    f"time={result['time_s']:.1f}s final={result['final_distance_m']:.1f}m",
                    flush=True,
                )
    finally:
        env.close()

    csv_path = csv_dir / "evaluation_metrics.csv"
    summary = write_metrics(csv_path, rows)
    write_summary(csv_dir / "evaluation_summary.csv", summary)
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    png_path = plot_summary(summary, args.output)
    for row in summary:
        print(
            f"speed={row['target_speed_knots']:g} kn: "
            f"success={row['success_rate']:.0%}, collisions={row['collision_rate']:.0%}, "
            f"oob={row['out_of_bounds_rate']:.0%}, timeout={row['timeout_rate']:.0%}"
        )
    print(f"Saved {csv_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
