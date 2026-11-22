"""Collect teacher corrections on states visited by the current BC policy."""
import multiprocessing as mp

import numpy as np
import torch
from stable_baselines3 import PPO

from kinematic_uuv.env import KNOT_MPS, KinematicUUVEnv
from kinematic_uuv.teacher import teacher_action
from training.teacher_data import compact_student_observation


def collect_dagger_episode(job):
    model_path, speed_knots, seed, episode_steps, env_kwargs, deterministic = job
    torch.set_num_threads(1)
    model = PPO.load(model_path, device="cpu")
    env = KinematicUUVEnv(
        max_steps=episode_steps,
        **(env_kwargs or {}),
    )
    env.cruise_speed_override = float(speed_knots) * KNOT_MPS
    observation, _ = env.reset(seed=int(seed))
    observations = []
    actions = []
    info = {}
    try:
        while True:
            teacher = teacher_action(
                observation,
                path=env.teacher_path_observation(),
            )
            observations.append(compact_student_observation(observation))
            actions.append(np.asarray(teacher, dtype=np.float32).copy())
            student, _ = model.predict(observation, deterministic=deterministic)
            observation, _, terminated, truncated, info = env.step(student)
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "observations": {
            key: np.stack([row[key] for row in observations])
            for key in observations[0]
        },
        "actions": np.asarray(actions, dtype=np.float32),
        "success": bool(info.get("reached", False)),
        "collision": bool(info.get("collision", False)),
        "out_of_bounds": bool(info.get("out_of_bounds", False)),
        "timeout": bool(
            not info.get("reached", False)
            and not info.get("collision", False)
            and not info.get("out_of_bounds", False)
        ),
    }


def collect_dagger_dataset(model_path, target_steps, seed_start, speeds_knots,
                           episode_steps=800, workers=4, env_kwargs=None,
                           deterministic=True):
    speeds_knots = tuple(float(value) for value in speeds_knots)
    if not speeds_knots:
        raise ValueError("at least one DAgger speed is required")
    workers = max(1, int(workers))
    batch_size = max(4, workers * 2)
    results = []
    collected = 0
    attempts = 0
    max_attempts = max(100, (target_steps // 20) * 4)
    outcomes = {"success": 0, "collision": 0, "out_of_bounds": 0, "timeout": 0}
    context = mp.get_context("spawn")
    pool = context.Pool(workers) if workers > 1 else None
    try:
        while collected < target_steps:
            if attempts >= max_attempts:
                raise RuntimeError(
                    "DAgger collection could not reach its target"
                )
            jobs = [
                (
                    str(model_path),
                    speeds_knots[(attempts + index) % len(speeds_knots)],
                    seed_start + attempts + index,
                    episode_steps,
                    env_kwargs or {},
                    bool(deterministic),
                )
                for index in range(batch_size)
            ]
            if pool is None:
                batch = [collect_dagger_episode(job) for job in jobs]
            else:
                batch = pool.map(collect_dagger_episode, jobs)
            attempts += len(jobs)
            for episode in batch:
                results.append(episode)
                collected += len(episode["actions"])
                for key in outcomes:
                    outcomes[key] += int(episode[key])
            print(
                f"DAgger collection: {collected}/{target_steps} steps, "
                f"{len(results)} episodes, {attempts} attempts",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return {
        "observations": {
            key: np.concatenate([episode["observations"][key] for episode in results])
            for key in results[0]["observations"]
        },
        "actions": np.concatenate([episode["actions"] for episode in results]),
        "steps": collected,
        "episodes": len(results),
        "attempts": attempts,
        "outcomes": outcomes,
    }
