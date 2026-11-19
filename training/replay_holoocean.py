"""Replay a kinematic policy trajectory in a HoloOcean scene for geometry checks."""
import argparse
import time
from pathlib import Path

import numpy as np


def _scenario(start_position_nwu):
    return {
        "name": "kinematic_uuv_replay",
        "package_name": "Ocean",
        "world": "OpenWater",
        "main_agent": "auv0",
        "ticks_per_sec": 10,
        "frames_per_sec": False,
        "agents": [{
            "agent_name": "auv0",
            "agent_type": "TorpedoAUV",
            "control_scheme": 1,
            "fossen_model": "torpedo",
            "control_mode": "manualControl",
            "location": [float(v) for v in start_position_nwu],
            "rotation": [0.0, 0.0, 0.0],
            "sensors": [],
        }],
    }


def _ned_to_nwu(position, pitch, yaw):
    location = np.array([position[0], -position[1], -position[2]], dtype=float)
    rotation = np.rad2deg([0.0, -pitch, -yaw])
    return location, rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path,
                        help=".npz trajectory exported by training.evaluate")
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=2.0,
                        help="keep the final frame visible before closing")
    args = parser.parse_args()
    if not args.trajectory.is_file():
        parser.error(f"trajectory does not exist: {args.trajectory}")

    data = np.load(args.trajectory)
    positions = data["position"]
    rotations = data["roll_pitch_yaw"]
    obstacles = data["obstacles"]
    goal = data["goal"]
    goal_radius = float(data["goal_radius_m"]) if "goal_radius_m" in data else 5.0
    speed = data["speed"]
    success = bool(data["success"])
    speed_knots = float(data["target_speed_knots"])
    start_location, _ = _ned_to_nwu(positions[0], rotations[0, 1], rotations[0, 2])

    import holoocean

    world = holoocean.make(
        scenario_cfg=_scenario(start_location),
        show_viewport=not args.headless,
        verbose=False,
    )
    try:
        world.set_render_quality(0)
        if not args.headless:
            world.should_render_viewport(True)
        for index, values in enumerate(obstacles):
            center = values[:3]
            radius = float(values[3])
            world.spawn_prop(
                "sphere",
                location=[float(center[0]), -float(center[1]), -float(center[2])],
                scale=2.0 * radius,
                sim_physics=False,
                material="gold",
                tag=f"kinematic_obstacle_{index}",
            )
        world.spawn_prop(
            "sphere",
            location=[float(goal[0]), -float(goal[1]), -float(goal[2])],
            scale=2.0 * goal_radius,
            sim_physics=False,
            material="white",
            tag="kinematic_goal",
        )
        agent = world.agents["auv0"]
        clearance = float("inf")
        print(
            f"Replaying {len(positions)} frames at target speed {speed_knots:g} kn; "
            f"success={success}",
            flush=True,
        )
        for index, position in enumerate(positions):
            pitch = float(rotations[index, 1])
            yaw = float(rotations[index, 2])
            location, rotation = _ned_to_nwu(position, pitch, yaw)
            agent.teleport(location=location, rotation=rotation)
            world.tick()
            point_clearance = min(
                (np.linalg.norm(position - values[:3]) - float(values[3])
                 for values in obstacles),
                default=float("inf"),
            )
            clearance = min(clearance, point_clearance)
            if args.realtime:
                time.sleep(0.08)
            if index % 50 == 0:
                print(
                    f"frame={index} NED={position.round(2)} "
                    f"speed={float(speed[index]):.2f} m/s clearance={point_clearance:.2f} m",
                    flush=True,
                )
        print(
            f"Replay finished: min geometric clearance={clearance:.2f} m; "
            f"target success={success}",
            flush=True,
        )
        if args.hold_seconds > 0:
            time.sleep(args.hold_seconds)
    finally:
        world.__exit__(None, None, None)


if __name__ == "__main__":
    main()
