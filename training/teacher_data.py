"""Parallel collection of successful teacher trajectories for behavior cloning."""
import multiprocessing as mp
import os

import numpy as np

from kinematic_uuv.env import KNOT_MPS, KinematicUUVEnv
from kinematic_uuv.teacher import teacher_action


def compact_student_observation(observation):
    """Drop the redundant current frame when history is available."""
    skip_current_sonar = "sonar_history" in observation
    return {
        key: np.asarray(
            value,
            dtype=np.float16 if key == "sonar_history" else np.float32,
        ).copy()
        for key, value in observation.items()
        if not (skip_current_sonar and key == "sonar")
    }


def collect_teacher_episode(job):
    """Collect one complete episode, returning None unless the teacher succeeds."""
    if len(job) == 4:
        speed_knots, seed, episode_steps, env_kwargs = job
    else:
        speed_knots, seed, episode_steps = job
        env_kwargs = None
    env = KinematicUUVEnv(max_steps=episode_steps, **(env_kwargs or {}))
    env.cruise_speed_override = float(speed_knots) * KNOT_MPS
    observation, _ = env.reset(seed=int(seed))
    observations = []
    actions = []
    info = {}
    try:
        while True:
            action = teacher_action(
                observation,
                path=env.teacher_path_observation(),
            )
            observations.append(compact_student_observation(observation))
            actions.append(np.asarray(action, dtype=np.float32).copy())
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    if not info.get("reached", False):
        return None
    return {
        "seed": int(seed),
        "speed_knots": float(speed_knots),
        "observations": {
            key: np.stack([row[key] for row in observations])
            for key in observations[0]
        },
        "actions": np.asarray(actions, dtype=np.float32),
    }


def collect_teacher_dataset(target_steps, seed_start, speeds_knots,
                            episode_steps=800, workers=None, env_kwargs=None):
    """Collect successful episodes until roughly target_steps transitions."""
    speeds_knots = tuple(float(value) for value in speeds_knots)
    if not speeds_knots:
        raise ValueError("at least one teacher speed is required")
    if target_steps <= 0:
        return {
            "observations": {},
            "actions": np.empty((0, 3), dtype=np.float32),
            "steps": 0,
            "episodes": 0,
            "attempts": 0,
        }
    if workers is None:
        workers = min(32, max(1, (os.cpu_count() or 2) // 4))
    workers = max(1, int(workers))

    results = []
    collected = 0
    attempts = 0
    batch_size = max(workers * 2, 4)
    max_attempts = max(100, (target_steps // 20) * 4)
    context = mp.get_context("spawn")
    pool = context.Pool(workers) if workers > 1 else None
    try:
        while collected < target_steps:
            if attempts >= max_attempts:
                raise RuntimeError(
                    "teacher collection could not reach its success target"
                )
            jobs = [
                (
                    speeds_knots[(attempts + index) % len(speeds_knots)],
                    seed_start + attempts + index,
                    episode_steps,
                    env_kwargs or {},
                )
                for index in range(batch_size)
            ]
            if pool is None:
                batch = [collect_teacher_episode(job) for job in jobs]
            else:
                batch = pool.map(collect_teacher_episode, jobs)
            attempts += len(jobs)
            for episode in batch:
                if episode is None:
                    continue
                results.append(episode)
                collected += len(episode["actions"])
            print(
                f"BC collection: {collected}/{target_steps} successful steps, "
                f"{len(results)} episodes, {attempts} attempts",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    observations = {
        key: np.concatenate([episode["observations"][key] for episode in results])
        for key in results[0]["observations"]
    }
    return {
        "observations": observations,
        "actions": np.concatenate([episode["actions"] for episode in results]),
        "steps": collected,
        "episodes": len(results),
        "attempts": attempts,
    }
