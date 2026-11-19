"""Path-following teacher used only for behavior-cloning initialization."""
import numpy as np


def teacher_action(observation):
    path_yaw = float(observation["path"][1])
    path_pitch = float(observation["path"][0])
    cross_track = float(observation["path"][4])
    target_pitch = float(observation["target"][0])
    target_yaw = float(observation["target"][1])
    remaining_distance = float(observation["target"][3]) * 500.0
    motion = observation["motion"]
    speed, cruise, speed_min, speed_max = map(float, motion[:4])
    yaw_rate = float(motion[4])
    pitch_rate = float(motion[6])

    terminal_distance = max(30.0, 2.5 * speed)
    terminal_weight = float(np.clip(
        (terminal_distance - remaining_distance) / terminal_distance,
        0.0, 1.0,
    ))
    yaw_error = (
        (1.0 - terminal_weight) * path_yaw
        + terminal_weight * target_yaw
    )
    pitch_error = (
        (1.0 - terminal_weight) * path_pitch
        + terminal_weight * target_pitch
    )
    turn_factor = max(0.35, float(np.cos(path_yaw)))
    sonar = observation["sonar"]
    front_hits = sonar[1] > 0.5
    front_distance = sonar[0]
    near_front = bool(np.any(front_hits[:, 16:32]
                             & (front_distance[:, 16:32] < 50.0 / 150.0)))
    side_hits = (sonar[[3, 5]] > 0.5) & (sonar[[2, 4]] < 25.0 / 150.0)
    if near_front or np.any(side_hits):
        turn_factor *= 0.55
    desired_speed = cruise * turn_factor

    # Start reducing speed early enough for the environment's 3 m/s^2 braking
    # limit, then converge on the actual goal point in the terminal phase.
    stopping_distance = speed * speed / 6.0 + 10.0
    if remaining_distance < stopping_distance:
        stopping_speed = np.sqrt(max(
            0.0, 6.0 * max(0.0, remaining_distance - 5.0)
        ))
        desired_speed = min(desired_speed, stopping_speed)
    desired_speed = float(np.clip(
        desired_speed, speed_min, speed_max
    ))
    speed_action = 2.0 * (desired_speed - speed_min) / max(
        speed_max - speed_min, 1e-6
    ) - 1.0
    return np.array([
        np.clip(speed_action, -1.0, 1.0),
        np.clip(
            2.0 * yaw_error
            - 1.2 * (1.0 - terminal_weight) * cross_track
            - 0.25 * yaw_rate,
            -1.0, 1.0,
        ),
        np.clip(3.0 * pitch_error - 0.35 * pitch_rate, -1.0, 1.0),
    ], dtype=np.float32)
