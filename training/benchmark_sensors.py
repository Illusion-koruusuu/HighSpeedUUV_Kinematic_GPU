"""Compare CPU and GPU sensor backends using identical kinematic scenes."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from kinematic_uuv.env import KinematicUUVEnv
from kinematic_uuv.teacher import teacher_action


def benchmark_backend(backend, sensor_device, min_obstacles, max_obstacles,
                      steps, warmup, seed):
    env = KinematicUUVEnv(
        max_steps=steps + warmup + 10,
        obstacle_count_range=(min_obstacles, max_obstacles),
        sensor_backend=backend,
        sensor_device=sensor_device,
    )
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(warmup):
            observation, _, terminated, truncated, _ = env.step(
                teacher_action(observation)
            )
            if terminated or truncated:
                observation, _ = env.reset(seed=seed)
        use_cuda = backend == "gpu" and torch.device(sensor_device).type == "cuda"
        if use_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        completed = 0
        for _ in range(steps):
            observation, _, terminated, truncated, _ = env.step(
                teacher_action(observation)
            )
            completed += 1
            if terminated or truncated:
                observation, _ = env.reset(seed=seed)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        env.close()
    return {
        "sensor_backend": backend,
        "sensor_device": sensor_device if backend == "gpu" else "cpu",
        "obstacles": len(env.obstacles),
        "steps": completed,
        "elapsed_s": elapsed,
        "steps_per_s": completed / max(elapsed, 1e-9),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-obstacles", type=int, default=100)
    parser.add_argument("--max-obstacles", type=int, default=150)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--sensor-device", default="cuda")
    parser.add_argument("--output", type=Path,
                        default=Path("runs/sensor_benchmark"))
    args = parser.parse_args()
    if min(args.steps, args.warmup) < 0 or args.min_obstacles < 0:
        parser.error("steps, warmup and obstacle counts must be nonnegative")
    if args.max_obstacles < args.min_obstacles:
        parser.error("invalid obstacle range")

    results = []
    for backend in ("cpu", "gpu"):
        print(f"Benchmarking sensor backend: {backend}", flush=True)
        results.append(benchmark_backend(
            backend, args.sensor_device, args.min_obstacles,
            args.max_obstacles, args.steps, args.warmup, args.seed,
        ))
    cpu_fps = results[0]["steps_per_s"]
    for result in results:
        result["speedup_vs_cpu"] = result["steps_per_s"] / max(cpu_fps, 1e-9)
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "sensor_benchmark.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output / "sensor_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for result in results:
        print(
            f"{result['sensor_backend']}: {result['steps_per_s']:.2f} steps/s, "
            f"speedup={result['speedup_vs_cpu']:.2f}x"
        )
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()

