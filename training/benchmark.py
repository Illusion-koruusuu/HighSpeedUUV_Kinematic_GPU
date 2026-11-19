"""Benchmark speed tracking, acceleration, turning and braking constraints."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kinematic_uuv.env import KNOT_MPS, KinematicUUVEnv


def _speed_action(env, target_speed):
    return 2.0 * (target_speed - env.speed_min) / (
        env.speed_max - env.speed_min
    ) - 1.0


def _clear_obstacles(env):
    env.obstacles = []
    env.goal = np.array([650.0, 0.0, 50.0], dtype=float)
    env._build_cost_field()


def _run_acceleration(env, target_speed, max_seconds):
    observation, _ = env.reset(seed=0)
    _clear_obstacles(env)
    start = env.state["position"].copy()
    speed_action = _speed_action(env, target_speed)
    speed_history = []
    time_to_90 = None
    distance_to_90 = None
    steps = int(max_seconds / env.dt)
    for step in range(1, steps + 1):
        observation, _, terminated, truncated, _ = env.step(
            np.array([speed_action, 0.0, 0.0], dtype=np.float32)
        )
        speed_history.append(env.state["speed"])
        if time_to_90 is None and env.state["speed"] >= 0.9 * target_speed:
            time_to_90 = step * env.dt
            distance_to_90 = float(np.linalg.norm(
                env.state["position"] - start
            ))
        if terminated or truncated:
            break
    return {
        "time_to_90pct_s": time_to_90,
        "final_speed_mps": float(env.state["speed"]),
        "acceleration_distance_m": distance_to_90,
        "max_speed_mps": float(np.max(speed_history)),
    }


def _run_turn(env, target_speed, max_seconds):
    observation, _ = env.reset(seed=0)
    _clear_obstacles(env)
    speed_action = _speed_action(env, target_speed)
    for _ in range(int(max_seconds / env.dt)):
        observation, _, terminated, truncated, _ = env.step(
            np.array([speed_action, 0.0, 0.0], dtype=np.float32)
        )
        if env.state["speed"] >= 0.98 * target_speed:
            break
    start = env.state["position"].copy()
    lat_accels = []
    yaw_rates = []
    speeds = []
    for _ in range(int(5.0 / env.dt)):
        observation, _, terminated, truncated, _ = env.step(
            np.array([speed_action, 1.0, 0.0], dtype=np.float32)
        )
        speeds.append(env.state["speed"])
        yaw_rates.append(env.state["yaw_rate"])
        lat_accels.append(env.state["speed"] * abs(env.state["yaw_rate"]))
        if terminated or truncated:
            break
    mean_speed = float(np.mean(speeds))
    mean_yaw_rate = float(np.mean(np.abs(yaw_rates)))
    return {
        "turn_mean_speed_mps": mean_speed,
        "turn_radius_m": (mean_speed / mean_yaw_rate
                          if mean_yaw_rate > 1e-8 else None),
        "max_lat_accel_mps2": float(np.max(lat_accels)),
        "max_yaw_rate_rad_s": float(np.max(np.abs(yaw_rates))),
        "turn_displacement_m": float(np.linalg.norm(
            env.state["position"] - start
        )),
    }


def _run_braking(env, target_speed, max_seconds):
    observation, _ = env.reset(seed=0)
    _clear_obstacles(env)
    speed_action = _speed_action(env, target_speed)
    for _ in range(int(max_seconds / env.dt)):
        observation, _, terminated, truncated, _ = env.step(
            np.array([speed_action, 0.0, 0.0], dtype=np.float32)
        )
        if env.state["speed"] >= 0.98 * target_speed:
            break
    initial_speed = env.state["speed"]
    start = env.state["position"].copy()
    braking_time = None
    for step in range(1, int(max_seconds / env.dt) + 1):
        observation, _, terminated, truncated, _ = env.step(
            np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        )
        if env.state["speed"] <= env.speed_min + 0.05:
            braking_time = step * env.dt
            break
        if terminated or truncated:
            break
    return {
        "brake_initial_speed_mps": float(initial_speed),
        "braking_time_s": braking_time,
        "braking_distance_m": float(np.linalg.norm(
            env.state["position"] - start
        )),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds-knots", type=float, nargs="+",
                        default=[10.0, 20.0, 30.0, 40.0, 50.0])
    parser.add_argument("--output", type=Path,
                        default=Path("output_model/kinematic_benchmark"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    csv_dir = args.output / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    env = KinematicUUVEnv(max_steps=2_000)
    rows = []
    try:
        for speed_knots in args.speeds_knots:
            target = speed_knots * KNOT_MPS
            result = {"target_speed_knots": speed_knots}
            result.update(_run_acceleration(env, target, 30.0))
            result.update(_run_turn(env, target, 30.0))
            result.update(_run_braking(env, target, 40.0))
            rows.append(result)
            print(
                f"{speed_knots:g} kn: reach={result['final_speed_mps']:.2f} m/s, "
                f"0-90%={result['time_to_90pct_s']:.2f}s, "
                f"turn_R={result['turn_radius_m']:.1f}m, "
                f"lat={result['max_lat_accel_mps2']:.2f}m/s2, "
                f"brake={result['braking_distance_m']:.1f}m",
                flush=True,
            )
    finally:
        env.close()

    json_path = args.output / "kinematic_benchmark.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    csv_path = csv_dir / "kinematic_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    speeds = [row["target_speed_knots"] for row in rows]
    fig, (ax_speed, ax_turn) = plt.subplots(1, 2, figsize=(12, 5))
    ax_speed.plot(speeds, [row["final_speed_mps"] / KNOT_MPS for row in rows],
                  marker="o", label="reached")
    ax_speed.plot(speeds, speeds, linestyle="--", label="target")
    ax_speed.set_xlabel("Target speed (knots)")
    ax_speed.set_ylabel("Speed (knots)")
    ax_speed.set_title("Speed tracking")
    ax_speed.grid(True)
    ax_speed.legend()
    ax_turn.plot(speeds, [row["turn_radius_m"] for row in rows],
                 marker="o", label="turn radius")
    ax_turn.set_xlabel("Target speed (knots)")
    ax_turn.set_ylabel("Turn radius (m)")
    ax_turn.set_title("Constrained turning")
    ax_turn.grid(True)
    ax_turn.legend()
    fig.tight_layout()
    png_path = args.output / "kinematic_benchmark.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
