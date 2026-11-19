"""Speed-constrained three-dimensional kinematic UUV environment."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dijkstra import build_cost_field, interpolate_cost
from .sensors import SonarArray, TorchSonarArray


KNOT_MPS = 0.5144444444


class KinematicUUVEnv(gym.Env):
    """Kinematic UUV with speed, acceleration, turn-radius and pitch limits."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt=0.1,
        max_steps=800,
        speed_min=1.0,
        speed_max=50.0 * KNOT_MPS,
        cruise_speed_range=(5.0 * KNOT_MPS, 50.0 * KNOT_MPS),
        max_accel=2.5,
        max_decel=3.0,
        max_yaw_rate=0.8,
        max_yaw_accel=1.0,
        min_yaw_rate=0.08,
        max_lat_accel=6.0,
        min_turn_radius=20.0,
        max_pitch_rate=0.20,
        max_pitch=0.35,
        goal_distance_range=(120.0, 450.0),
        obstacle_count_range=(20, 32),
        obstacle_radius_range=(4.0, 22.0),
        obstacle_distribution="uniform",
        goal_radius=5.0,
        planning_bounds=((-100.0, -350.0, 0.0),
                         (700.0, 350.0, 100.0)),
        dijkstra_resolution=8.0,
        vertical_cost_scale=2.0,
        path_refresh_steps=5,
        sensor_backend="cpu",
        sensor_device="cuda",
    ):
        super().__init__()
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.cruise_speed_range = tuple(float(v) for v in cruise_speed_range)
        self.max_accel = float(max_accel)
        self.max_decel = float(max_decel)
        self.max_yaw_rate = float(max_yaw_rate)
        self.max_yaw_accel = float(max_yaw_accel)
        self.min_yaw_rate = float(min_yaw_rate)
        self.max_lat_accel = float(max_lat_accel)
        self.min_turn_radius = float(min_turn_radius)
        self.max_pitch_rate = float(max_pitch_rate)
        self.max_pitch = float(max_pitch)
        self.goal_distance_range = tuple(float(v) for v in goal_distance_range)
        self.obstacle_count_min, self.obstacle_count_max = map(
            int, obstacle_count_range
        )
        self.obstacle_radius_min, self.obstacle_radius_max = map(
            float, obstacle_radius_range
        )
        if self.obstacle_radius_min <= 0.0 or (
                self.obstacle_radius_max < self.obstacle_radius_min):
            raise ValueError("invalid obstacle radius range")
        self.obstacle_distribution = str(obstacle_distribution)
        if self.obstacle_distribution != "uniform":
            raise ValueError("only uniform obstacle distribution is supported")
        self.goal_radius = float(goal_radius)
        self.planning_bounds = (
            np.asarray(planning_bounds[0], dtype=float),
            np.asarray(planning_bounds[1], dtype=float),
        )
        self.dijkstra_resolution = float(dijkstra_resolution)
        self.vertical_cost_scale = float(vertical_cost_scale)
        self.path_refresh_steps = max(1, int(path_refresh_steps))
        self.sensor_backend = str(sensor_backend)
        self.sensor_device = str(sensor_device)
        if self.sensor_backend == "gpu":
            self.sonar = TorchSonarArray(device=self.sensor_device)
        elif self.sensor_backend == "cpu":
            self.sonar = SonarArray()
        else:
            raise ValueError("sensor_backend must be 'cpu' or 'gpu'")

        self.state = {
            "position": np.zeros(3, dtype=float),
            "yaw": 0.0,
            "pitch": 0.0,
            "speed": 0.0,
            "yaw_rate": 0.0,
            "pitch_rate": 0.0,
        }
        self.goal = np.zeros(3, dtype=float)
        self.obstacles = []
        self.cruise_speed = float(np.mean(self.cruise_speed_range))
        self.cruise_speed_override = None
        self.steps = 0
        self._cost_field = None
        self._cost_origin = None
        self._cost_occupancy = None
        self._cached_path = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "sonar": spaces.Box(0.0, 1.0, shape=self.sonar.observation_shape,
                               dtype=np.float32),
            "target": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "path": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
            "motion": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
        })

    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    @staticmethod
    def _segment_distance(point, start, end):
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared <= 1e-12:
            return float(np.linalg.norm(point - start))
        fraction = float(np.clip(
            np.dot(point - start, segment) / length_squared, 0.0, 1.0
        ))
        return float(np.linalg.norm(point - (start + fraction * segment)))

    def set_cruise_speed_range(self, speed_range):
        self.cruise_speed_range = tuple(float(v) for v in speed_range)

    def _outside_bounds(self, position):
        lower, upper = self.planning_bounds
        return bool(np.any(position < lower) or np.any(position > upper))

    def _sample_scene(self):
        start = np.array([0.0, 0.0, 10.0], dtype=float)
        distance = self.np_random.uniform(*self.goal_distance_range)
        yaw = self.np_random.uniform(-np.pi / 4.0, np.pi / 4.0)
        pitch = self.np_random.uniform(-np.pi / 12.0, np.pi / 12.0)
        self.goal = np.array([
            distance * np.cos(pitch) * np.cos(yaw),
            distance * np.cos(pitch) * np.sin(yaw),
            np.clip(start[2] - distance * np.sin(pitch), 8.0, 75.0),
        ])
        if self.cruise_speed_override is None:
            self.cruise_speed = float(self.np_random.uniform(
                *self.cruise_speed_range
            ))
        else:
            self.cruise_speed = float(self.cruise_speed_override)
        count = int(self.np_random.integers(
            self.obstacle_count_min, self.obstacle_count_max + 1
        ))
        for _attempt in range(20):
            obstacles = self._sample_obstacles(start, count)
            if len(obstacles) == count:
                self.obstacles = obstacles
                return
        raise RuntimeError("failed to generate a valid obstacle scene")

    def _sample_radius(self):
        draw = float(self.np_random.random())
        if draw < 0.45:
            low, high = self.obstacle_radius_min, min(
                8.0, self.obstacle_radius_max
            )
        elif draw < 0.75:
            low, high = max(8.0, self.obstacle_radius_min), min(
                14.0, self.obstacle_radius_max
            )
        else:
            low, high = max(14.0, self.obstacle_radius_min), \
                self.obstacle_radius_max
        return float(self.np_random.uniform(low, max(low, high)))

    def _sample_obstacles(self, start, count):
        lower, upper = self.planning_bounds
        obstacles = []
        for _obstacle_index in range(count):
            placed = False
            for _attempt in range(600):
                radius = self._sample_radius()
                center = self.np_random.uniform(lower + 2.0, upper - 2.0)
                separated = all(
                    np.linalg.norm(center - other_center)
                    > radius + other_radius + 2.0
                    for other_center, other_radius in obstacles
                )
                clears_start = (
                    np.linalg.norm(center - start) > radius + 26.0
                )
                clears_goal = (
                    np.linalg.norm(center - self.goal) > radius + 22.0
                )
                if clears_start and clears_goal and separated:
                    obstacles.append((center, radius))
                    placed = True
                    break
            if not placed:
                return []
        return obstacles

    def _build_cost_field(self):
        self._cost_field, self._cost_origin, self.dijkstra_resolution, \
            self._cost_occupancy = build_cost_field(
                self.goal,
                self.obstacles,
                self.state["position"],
                resolution=self.dijkstra_resolution,
                obstacle_margin=15.0,
                goal_radius=self.goal_radius,
                planner_goal_radius=0.0,
                planning_bounds=self.planning_bounds,
                vertical_cost_scale=self.vertical_cost_scale,
            )

    def _dijkstra_cost(self, position):
        if self._cost_field is None:
            return float("inf")
        cost = interpolate_cost(
            self._cost_field, self._cost_origin,
            self.dijkstra_resolution, position,
        )
        if np.isfinite(cost):
            return cost
        finite = self._cost_field[np.isfinite(self._cost_field)]
        max_cost = float(finite.max()) if finite.size else 0.0
        upper = self._cost_origin + (
            np.asarray(self._cost_field.shape) - 1.0
        ) * self.dijkstra_resolution
        outside = np.maximum(self._cost_origin - position, 0.0)
        outside += np.maximum(position - upper, 0.0)
        return max_cost + float(np.linalg.norm(outside))

    def _plan_path(self, start, max_points=500):
        if self._cost_field is None:
            return np.empty((0, 3), dtype=float)
        shape = np.asarray(self._cost_field.shape, dtype=int)
        current = tuple(np.rint(
            (np.asarray(start) - self._cost_origin) / self.dijkstra_resolution
        ).astype(int).clip(0, shape - 1))
        path = [self._cost_origin + np.asarray(current) * self.dijkstra_resolution]
        offsets = [
            (dx, dy, dz) for dx in (-1, 0, 1)
            for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        ]
        for _ in range(max_points):
            current_cost = float(self._cost_field[current])
            if not np.isfinite(current_cost) or current_cost <= 1e-5:
                break
            candidates = []
            for offset in offsets:
                neighbor = tuple(current[i] + offset[i] for i in range(3))
                if any(index < 0 or index >= shape[axis]
                       for axis, index in enumerate(neighbor)):
                    continue
                if self._cost_occupancy[neighbor]:
                    continue
                neighbor_cost = float(self._cost_field[neighbor])
                if np.isfinite(neighbor_cost):
                    candidates.append((neighbor_cost, neighbor))
            if not candidates:
                break
            next_cost, next_index = min(candidates, key=lambda item: item[0])
            if next_cost >= current_cost - 1e-6:
                break
            current = next_index
            path.append(self._cost_origin + np.asarray(current)
                        * self.dijkstra_resolution)
        return np.asarray(path, dtype=float)

    def _path_observation(self, path):
        if len(path) < 2:
            return np.zeros(5, dtype=np.float32)
        point = self.state["position"]
        best_error = float("inf")
        closest = path[0]
        tangent = path[1] - path[0]
        start = 0
        for index, (segment_start, segment_end) in enumerate(zip(path[:-1], path[1:])):
            segment = segment_end - segment_start
            length_squared = float(np.dot(segment, segment))
            if length_squared <= 1e-12:
                continue
            fraction = float(np.clip(
                np.dot(point - segment_start, segment) / length_squared, 0.0, 1.0
            ))
            candidate = segment_start + fraction * segment
            error = float(np.linalg.norm(point - candidate))
            if error < best_error:
                best_error = error
                closest = candidate
                tangent = segment
                start = index
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > 1e-9:
            tangent = tangent / tangent_norm
        lookahead = path[-1]
        lookahead_distance = 25.0 + 2.0 * self.state["speed"]
        for candidate in path[start:]:
            if np.linalg.norm(candidate - point) >= lookahead_distance:
                lookahead = candidate
                break
        delta = lookahead - point
        horizontal = float(np.hypot(delta[0], delta[1]))
        pitch = float(np.arctan2(-delta[2], max(horizontal, 1e-6))
                      - self.state["pitch"])
        yaw = float(self._wrap_angle(
            np.arctan2(delta[1], delta[0]) - self.state["yaw"]
        ))
        remaining = self._dijkstra_cost(point)
        right = np.array([
            -np.sin(self.state["yaw"]), np.cos(self.state["yaw"]), 0.0
        ])
        cross_track = float(np.dot(point - closest, right))
        return np.array([
            pitch,
            yaw,
            float(np.linalg.norm(delta)) / 100.0,
            float(np.clip(remaining / 500.0, 0.0, 2.0)),
            float(np.clip(cross_track / 30.0, -1.0, 1.0)),
        ], dtype=np.float32)

    def _observe(self, path=None):
        eta = np.array([
            self.state["position"][0], self.state["position"][1],
            self.state["position"][2], 0.0, self.state["pitch"],
            self.state["yaw"],
        ])
        sonar = self.sonar.observe(eta, self.obstacles)
        delta = self.goal - self.state["position"]
        distance = float(np.linalg.norm(delta))
        horizontal = float(np.hypot(delta[0], delta[1]))
        target_pitch = float(np.arctan2(-delta[2], max(horizontal, 1e-6))
                             - self.state["pitch"])
        target_yaw = float(self._wrap_angle(
            np.arctan2(delta[1], delta[0]) - self.state["yaw"]
        ))
        target = np.array([
            target_pitch,
            target_yaw,
            self.cruise_speed - self.state["speed"],
            float(np.clip(distance / 500.0, 0.0, 1.0)),
        ], dtype=np.float32)
        path = self._current_path() if path is None else path
        motion = np.array([
            self.state["speed"],
            self.cruise_speed,
            self.speed_min,
            self.speed_max,
            self.state["yaw_rate"],
            self.state["pitch"],
            self.state["pitch_rate"],
        ], dtype=np.float32)
        return {
            "sonar": sonar,
            "target": target,
            "path": self._path_observation(path),
            "motion": motion,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = {
            "position": np.array([0.0, 0.0, 10.0], dtype=float),
            "yaw": 0.0,
            "pitch": 0.0,
            "speed": self.speed_min,
            "yaw_rate": 0.0,
            "pitch_rate": 0.0,
        }
        self.steps = 0
        for _attempt in range(10):
            self._sample_scene()
            self._build_cost_field()
            if np.isfinite(self._dijkstra_cost(self.state["position"])):
                break
        else:
            raise RuntimeError("failed to generate a reachable obstacle scene")
        self._cached_path = self._plan_path(self.state["position"])
        if hasattr(self.sonar, "set_obstacles"):
            self.sonar.set_obstacles(self.obstacles)
        return self._observe(), {}

    def _current_path(self):
        if (self._cached_path is None
                or self.steps % self.path_refresh_steps == 0):
            self._cached_path = self._plan_path(self.state["position"])
        return self._cached_path

    def _yaw_rate_limit(self):
        speed = max(self.state["speed"], 1.0)
        return min(
            self.max_yaw_rate,
            self.max_lat_accel / speed,
            max(self.min_yaw_rate, self.state["speed"] / self.min_turn_radius),
        )

    def _desired_speed(self, path, distance):
        yaw_error = abs(self._path_observation(path)[1])
        turn_factor = max(0.35, float(np.cos(yaw_error)))
        desired = self.cruise_speed * turn_factor
        brake_distance = 40.0 + self.state["speed"]
        if distance < brake_distance:
            desired = min(desired, max(self.speed_min,
                                      self.cruise_speed * distance / brake_distance))
        return float(np.clip(desired, self.speed_min, self.speed_max))

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        path = self._current_path()
        previous_observation = self._observe(path)
        previous_position = self.state["position"].copy()
        old_distance = float(np.linalg.norm(self.goal - previous_position))

        desired_speed = self.speed_min + 0.5 * (action[0] + 1.0) * (
            self.speed_max - self.speed_min
        )
        acceleration = float(np.clip(
            (desired_speed - self.state["speed"]) / self.dt,
            -self.max_decel, self.max_accel,
        ))
        self.state["speed"] = float(np.clip(
            self.state["speed"] + acceleration * self.dt,
            self.speed_min, self.speed_max,
        ))

        yaw_limit = self._yaw_rate_limit()
        target_yaw_rate = float(action[1] * yaw_limit)
        self.state["yaw_rate"] = float(np.clip(
            target_yaw_rate,
            self.state["yaw_rate"] - self.max_yaw_accel * self.dt,
            self.state["yaw_rate"] + self.max_yaw_accel * self.dt,
        ))
        self.state["yaw"] = float(self._wrap_angle(
            self.state["yaw"] + self.state["yaw_rate"] * self.dt
        ))

        target_pitch_rate = float(action[2] * self.max_pitch_rate)
        self.state["pitch_rate"] = float(np.clip(
            target_pitch_rate,
            self.state["pitch_rate"] - self.max_yaw_accel * self.dt,
            self.state["pitch_rate"] + self.max_yaw_accel * self.dt,
        ))
        self.state["pitch"] = float(np.clip(
            self.state["pitch"] + self.state["pitch_rate"] * self.dt,
            -self.max_pitch, self.max_pitch,
        ))
        if abs(self.state["pitch"]) >= self.max_pitch and np.sign(
                self.state["pitch_rate"]) == np.sign(self.state["pitch"]):
            self.state["pitch_rate"] = 0.0

        cos_pitch = float(np.cos(self.state["pitch"]))
        displacement = self.state["speed"] * self.dt * np.array([
            cos_pitch * np.cos(self.state["yaw"]),
            cos_pitch * np.sin(self.state["yaw"]),
            -np.sin(self.state["pitch"]),
        ])
        self.state["position"] += displacement
        self.steps += 1

        position = self.state["position"]
        distance = float(np.linalg.norm(self.goal - position))
        if hasattr(self.sonar, "segment_hits_and_clearance"):
            collided, clearance = self.sonar.segment_hits_and_clearance(
                previous_position, position, position
            )
        else:
            collided = any(
                self._segment_distance(center, previous_position, position) <= radius
                for center, radius in self.obstacles
            )
            clearance = min(
                (np.linalg.norm(position - center) - radius
                 for center, radius in self.obstacles),
                default=100.0,
            )
        reached = self._segment_distance(
            self.goal, previous_position, position
        ) <= self.goal_radius
        out_of_bounds = self._outside_bounds(position)

        cost_previous = self._dijkstra_cost(previous_position)
        cost_current = self._dijkstra_cost(position)
        if np.isfinite(cost_previous) and np.isfinite(cost_current):
            path_progress = float(np.clip(cost_previous - cost_current, -5.0, 5.0))
        else:
            path_progress = -1.0 if out_of_bounds else 0.0
        direct_progress = float(np.clip(old_distance - distance, -2.0, 2.0))
        observation = self._observe()
        alignment_error = abs(float(observation["path"][1]))
        desired_speed = self._desired_speed(path, distance)
        speed_error = abs(self.state["speed"] - desired_speed) / max(
            self.cruise_speed, 1.0
        )
        goal_proximity = 1.0 - float(np.clip(distance / 100.0, 0.0, 1.0))
        safety_penalty = max(0.0, 15.0 - clearance) / 15.0
        reward = (
            8.0 * path_progress
            + 2.0 * direct_progress
            - 0.5 * alignment_error
            - 0.3 * speed_error
            - 0.2 * safety_penalty
            - 0.02
            + 2.0 * goal_proximity
        )
        if reached:
            reward += 100.0
        if collided:
            reward -= 200.0
        if out_of_bounds:
            reward -= 200.0

        info = {
            "distance_to_goal": distance,
            "collision": bool(collided),
            "reached": bool(reached),
            "out_of_bounds": bool(out_of_bounds),
            "speed_mps": self.state["speed"],
            "cruise_speed_mps": self.cruise_speed,
            "yaw_rate_rad_s": self.state["yaw_rate"],
            "pitch_rad": self.state["pitch"],
            "clearance_m": float(clearance),
            "path_progress": path_progress,
            "alignment_error": alignment_error,
        }
        terminated = bool(collided or reached or out_of_bounds)
        truncated = bool(self.steps >= self.max_steps)
        if truncated and not terminated:
            reward -= 50.0
        self._previous_observation = previous_observation
        return observation, float(reward), terminated, truncated, info

    def close(self):
        pass
