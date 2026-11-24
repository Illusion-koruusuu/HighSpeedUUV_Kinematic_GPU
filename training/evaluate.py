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


def _make_env(config, episode_steps, moving_obstacles=False,
              obstacle_speed_range=(KNOT_MPS, 10.0 * KNOT_MPS),
              obstacle_turn_rate_rad_s=0.15):
    bounds = config.get("planning_bounds_m")
    obstacle_count_range = config.get("obstacle_count_range") or [0, 0]
    obstacle_radius_range = config.get(
        "obstacle_radius_range_m"
    ) or [4.0, 22.0]
    kwargs = dict(
        scene_type=config.get("scene_type", "spheres"),
        terrain_resolution=config.get("terrain_resolution_m", 20.0),
        terrain_peak_count_range=tuple(
            config.get("terrain_peak_count_range", [18, 30])
        ),
        terrain_base_height_range=tuple(
            config.get("terrain_base_height_range_m", [1.0, 7.0])
        ),
        terrain_max_height=config.get("terrain_max_height_m", 62.0),
        terrain_safety_margin=config.get("terrain_safety_margin_m", 8.0),
        terrain_goal_clearance=config.get(
            "terrain_goal_clearance_m", 12.0
        ),
        terrain_cruise_height=config.get(
            "terrain_cruise_height_m", 30.0
        ),
        include_depth_error_observation=config.get(
            "include_depth_error_observation",
            config.get("scene_type", "spheres") == "terrain",
        ),
        depth_hold_weight=config.get("depth_hold_weight", 0.2),
        depth_hold_scale=config.get("depth_hold_scale_m", 10.0),
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
        obstacle_count_range=tuple(obstacle_count_range),
        obstacle_radius_range=tuple(obstacle_radius_range),
        obstacle_distribution=config.get(
            "obstacle_distribution", "uniform"
        ),
        sensor_backend=config.get("sensor_backend", "cpu"),
        sensor_device=config.get("sensor_device", "cuda"),
        goal_distance_range=tuple(config.get("goal_distance_range_m", [120, 450])),
        dijkstra_resolution=config.get("dijkstra_resolution_m", 8.0),
        path_refresh_steps=config.get("path_refresh_steps", 5),
        dijkstra_potential_scale=config.get("dijkstra_potential_scale", 100.0),
        use_dijkstra=config.get("use_dijkstra", True),
        include_path_observation=config.get(
            "include_path_observation", True
        ),
        reward_mode=config.get("reward_mode", "dijkstra"),
        distance_potential_scale=config.get(
            "distance_potential_scale", 250.0
        ),
        reward_gamma=config.get("gamma", 0.999),
        obstacle_motion=(
            "random"
            if moving_obstacles and config.get("scene_type", "spheres")
            == "spheres"
            else "static"
        ),
        obstacle_speed_range=obstacle_speed_range,
        obstacle_turn_rate_rad_s=obstacle_turn_rate_rad_s,
        dijkstra_replan_steps=config.get("dijkstra_replan_steps", 0),
        sonar_history_length=config.get("sonar_history_length", 0),
        sonar_history_stride=config.get("sonar_history_stride", 1),
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
        obstacle_positions=np.asarray(record.get("obstacle_positions", [])),
        obstacles_initial=np.asarray(record.get("obstacles_initial", [])),
        obstacles_final=np.asarray(record.get("obstacles_final", [])),
        obstacle_motion=np.asarray(record.get("obstacle_motion", "static")),
        max_obstacle_speed_mps=np.asarray(
            record.get("max_obstacle_speed_mps", 0.0)
        ),
        scene_type=np.asarray(record.get("scene_type", "spheres")),
        terrain_height=np.asarray(record.get("terrain_height", [])),
        terrain_x=np.asarray(record.get("terrain_x", [])),
        terrain_y=np.asarray(record.get("terrain_y", [])),
    )


def run_episode(env, model, seed, speed_knots):
    env.cruise_speed_override = speed_knots * KNOT_MPS
    observation, _ = env.reset(seed=seed)
    position = [env.state["position"].copy()]
    rotation = [[0.0, env.state["pitch"], env.state["yaw"]]]
    speed = [env.state["speed"]]
    actions = []
    yaw_rate = [env.state["yaw_rate"]]
    terrain_meta = (
        env.terrain.metadata() if env.terrain is not None else {}
    )
    obstacles_initial = np.asarray([
        [*center, radius] for center, radius in env.obstacles
    ])
    obstacle_positions = [
        np.asarray([center for center, _ in env.obstacles], dtype=float)
    ]
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
        if env.obstacle_motion != "static":
            obstacle_positions.append(np.asarray(
                [center for center, _ in env.obstacles], dtype=float
            ))
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
        "obstacle_motion": env.obstacle_motion,
        "scene_type": env.scene_type,
        "max_obstacle_speed_mps": float(
            env.obstacle_speeds.max()
        ) if env.obstacle_speeds.size else 0.0,
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
        "obstacles_initial": obstacles_initial,
        "obstacles_final": np.asarray([
            [*center, radius] for center, radius in env.obstacles
        ]),
        "obstacle_positions": np.asarray(obstacle_positions),
        "obstacle_motion": env.obstacle_motion,
        "scene_type": env.scene_type,
        "max_obstacle_speed_mps": (
            float(env.obstacle_speeds.max())
            if env.obstacle_speeds.size else 0.0
        ),
        "cruise_speed_mps": env.cruise_speed,
        "success": bool(info.get("reached", False)),
        "seed": seed,
        "target_speed_knots": speed_knots,
        "dt": env.dt,
    }
    trajectory.update(terrain_meta)
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
    parser.add_argument("--episodes-per-speed", type=int, default=100)
    parser.add_argument("--speeds-knots", type=float, nargs="+",
                        default=[50.0])
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--save-trajectories", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device: cuda, cuda:0, cpu, or auto")
    parser.add_argument(
        "--moving-obstacles", action=argparse.BooleanOptionalAction,
        default=False,
        help="legacy spherical-scene obstacle motion",
    )
    parser.add_argument(
        "--obstacle-speed-min-knots", type=float, default=1.0,
        help="minimum obstacle speed in knots",
    )
    parser.add_argument(
        "--obstacle-speed-max-knots", type=float, default=10.0,
        help="maximum obstacle speed in knots",
    )
    parser.add_argument("--obstacle-turn-rate-rad-s", type=float, default=0.15)
    parser.add_argument("--dijkstra-replan-steps", type=int)
    args = parser.parse_args()
    if args.episodes_per_speed <= 0 or args.episode_steps <= 0:
        parser.error("episode counts must be positive")

    config = _load_config(args.model)
    observation_type = config.get("observation_type")
    if observation_type not in {
        "kinematic_sonar_path_v1",
        "kinematic_sonar_lstm_path_v2",
        "kinematic_sonar_goal_v2",
        "kinematic_sonar_lstm_goal_v3",
        "kinematic_sonar_lstm_seabed_v4",
    }:
        parser.error("model uses an unsupported observation configuration")
    device = resolve_device(args.device)
    configure_torch(device)
    if args.obstacle_speed_min_knots < 0.0 or (
            args.obstacle_speed_max_knots < args.obstacle_speed_min_knots):
        parser.error("invalid obstacle speed range")
    if args.obstacle_speed_max_knots > 50.0:
        parser.error("obstacle-speed-max-knots must not exceed 50 knots")
    if args.obstacle_speed_max_knots * KNOT_MPS > config.get(
            "speed_max_mps", 50.0 * KNOT_MPS):
        parser.error(
            "obstacle-speed-max-knots must not exceed UUV maximum speed"
        )
    if args.obstacle_turn_rate_rad_s < 0.0:
        parser.error("obstacle-turn-rate-rad-s must be nonnegative")
    if args.dijkstra_replan_steps is not None:
        if args.dijkstra_replan_steps < 0:
            parser.error("dijkstra-replan-steps must be nonnegative")
        config["dijkstra_replan_steps"] = args.dijkstra_replan_steps
    env = _make_env(
        config, args.episode_steps,
        moving_obstacles=args.moving_obstacles,
        obstacle_speed_range=(
            args.obstacle_speed_min_knots * KNOT_MPS,
            args.obstacle_speed_max_knots * KNOT_MPS,
        ),
        obstacle_turn_rate_rad_s=args.obstacle_turn_rate_rad_s,
    )
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
