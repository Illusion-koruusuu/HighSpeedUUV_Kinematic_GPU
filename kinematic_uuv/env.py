"""Speed-constrained three-dimensional kinematic UUV environment."""
from collections import deque

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
        goal_distance_range=(650.0, 900.0),
        obstacle_count_range=(20, 32),
        obstacle_radius_range=(4.0, 22.0),
        obstacle_distribution="uniform",
        goal_radius=5.0,
        planning_bounds=((-100.0, -750.0, 0.0),
                         (1100.0, 750.0, 100.0)),
        dijkstra_resolution=8.0,
        vertical_cost_scale=2.0,
        dijkstra_potential_scale=100.0,
        use_dijkstra=True,
        include_path_observation=True,
        reward_mode="dijkstra",
        distance_potential_scale=250.0,
        reward_gamma=0.999,
        path_refresh_steps=5,
        obstacle_motion="static",
        obstacle_speed_range=(1.0 * KNOT_MPS, 10.0 * KNOT_MPS),
        obstacle_turn_rate_rad_s=0.15,
        dijkstra_replan_steps=10,
        sonar_history_length=0,
        sonar_history_stride=1,
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
        self.dijkstra_potential_scale = float(dijkstra_potential_scale)
        if self.dijkstra_potential_scale <= 0.0:
            raise ValueError("dijkstra_potential_scale must be positive")
        self.use_dijkstra = bool(use_dijkstra)
        self.include_path_observation = bool(include_path_observation)
        if self.include_path_observation and not self.use_dijkstra:
            raise ValueError(
                "path observation requires Dijkstra guidance to be enabled"
            )
        self.reward_mode = str(reward_mode)
        if self.reward_mode not in {"dijkstra", "euclidean"}:
            raise ValueError("reward_mode must be 'dijkstra' or 'euclidean'")
        if self.reward_mode == "dijkstra" and not self.use_dijkstra:
            raise ValueError(
                "Dijkstra reward requires Dijkstra guidance to be enabled"
            )
        self.distance_potential_scale = float(distance_potential_scale)
        if self.distance_potential_scale <= 0.0:
            raise ValueError("distance_potential_scale must be positive")
        self.reward_gamma = float(reward_gamma)
        if not 0.0 < self.reward_gamma <= 1.0:
            raise ValueError("reward_gamma must be in (0, 1]")
        self.path_refresh_steps = max(1, int(path_refresh_steps))
        self.obstacle_motion = str(obstacle_motion)
        if self.obstacle_motion not in {"static", "random"}:
            raise ValueError("obstacle_motion must be 'static' or 'random'")
        self.obstacle_speed_min, self.obstacle_speed_max = map(
            float, obstacle_speed_range
        )
        if (self.obstacle_speed_min < 0.0
                or self.obstacle_speed_max < self.obstacle_speed_min):
            raise ValueError("invalid obstacle speed range")
        if self.obstacle_speed_max > self.speed_max:
            raise ValueError(
                "maximum obstacle speed must not exceed UUV maximum speed"
            )
        self.obstacle_turn_rate_rad_s = float(obstacle_turn_rate_rad_s)
        if self.obstacle_turn_rate_rad_s < 0.0:
            raise ValueError("obstacle turn rate must be nonnegative")
        self.dijkstra_replan_steps = int(dijkstra_replan_steps)
        if self.dijkstra_replan_steps < 0:
            raise ValueError("dijkstra_replan_steps must be nonnegative")
        self.sonar_history_length = int(sonar_history_length)
        if self.sonar_history_length < 0:
            raise ValueError("sonar_history_length must be nonnegative")
        self.sonar_history_stride = int(sonar_history_stride)
        if self.sonar_history_stride < 1:
            raise ValueError("sonar_history_stride must be positive")
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
        self.obstacle_velocities = np.empty((0, 3), dtype=float)
        self.obstacle_speeds = np.empty((0,), dtype=float)
        history_buffer_length = (
            (self.sonar_history_length - 1) * self.sonar_history_stride + 1
            if self.sonar_history_length > 0 else 1
        )
        self._sonar_history = deque(maxlen=history_buffer_length)
        self._cost_field = None
        self._cost_origin = None
        self._cost_occupancy = None
        self._cached_path = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        observation_spaces = {
            "sonar": spaces.Box(0.0, 1.0, shape=self.sonar.observation_shape,
                               dtype=np.float32),
            "target": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "motion": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
        }
        if self.include_path_observation:
            observation_spaces["path"] = spaces.Box(
                -np.inf, np.inf, shape=(5,), dtype=np.float32
            )
        if self.sonar_history_length > 0:
            observation_spaces["sonar_history"] = spaces.Box(
                0.0, 1.0,
                shape=(
                    self.sonar_history_length,
                    *self.sonar.observation_shape,
                ),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(observation_spaces)

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

    def _initialize_obstacle_motion(self):
        if self.obstacle_motion == "static" or not self.obstacles:
            self.obstacle_velocities = np.empty((0, 3), dtype=float)
            self.obstacle_speeds = np.empty((0,), dtype=float)
            return
        count = len(self.obstacles)
        self.obstacle_speeds = self.np_random.uniform(
            self.obstacle_speed_min,
            self.obstacle_speed_max,
            size=count,
        )
        directions = self.np_random.normal(size=(count, 3))
        norms = np.linalg.norm(directions, axis=1)
        norms = np.maximum(norms, 1e-9)
        directions /= norms[:, None]
        self.obstacle_velocities = directions * self.obstacle_speeds[:, None]

    def _update_obstacles(self):
        if self.obstacle_motion == "static" or not self.obstacles:
            return
        noise = self.np_random.normal(
            size=self.obstacle_velocities.shape
        ) * self.obstacle_turn_rate_rad_s * np.sqrt(self.dt)
        directions = self.obstacle_velocities + noise
        norms = np.linalg.norm(directions, axis=1)
        norms = np.maximum(norms, 1e-9)
        self.obstacle_velocities = (
            directions / norms[:, None]
        ) * self.obstacle_speeds[:, None]

        lower, upper = self.planning_bounds
        updated = []
        for index, (center, radius) in enumerate(self.obstacles):
            center = center + self.obstacle_velocities[index] * self.dt
            low = lower + radius
            high = upper - radius
            for axis in range(3):
                if center[axis] < low[axis]:
                    center[axis] = low[axis]
                    self.obstacle_velocities[index, axis] *= -1.0
                elif center[axis] > high[axis]:
                    center[axis] = high[axis]
                    self.obstacle_velocities[index, axis] *= -1.0
            updated.append((center, radius))
        self.obstacles = updated
        if hasattr(self.sonar, "set_obstacles"):
            self.sonar.set_obstacles(self.obstacles)

    def _sample_scene(self):
        start = np.array([0.0, 0.0, 10.0], dtype=float)
        distance = self.np_random.uniform(*self.goal_distance_range)
        yaw = self.np_random.uniform(-np.pi / 4.0, np.pi / 4.0)
        pitch = self.np_random.uniform(-np.pi / 12.0, np.pi / 12.0)
        depth = np.clip(start[2] - distance * np.sin(pitch), 8.0, 75.0)
        # Preserve the sampled 3D distance after limiting the goal depth.
        horizontal_distance = np.sqrt(distance**2 - (depth - start[2])**2)
        self.goal = np.array([
            horizontal_distance * np.cos(yaw),
            horizontal_distance * np.sin(yaw),
            depth,
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

    def _dijkstra_potential(self, position):
        """Potential is negative cost-to-go under the current cost field."""
        cost = self._dijkstra_cost(position)
        if not np.isfinite(cost):
            return float("-inf")
        return -cost / self.dijkstra_potential_scale

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

    def teacher_path_observation(self):
        """Return Dijkstra path features for teacher labels only."""
        if not self.use_dijkstra:
            return np.zeros(5, dtype=np.float32)
        return self._path_observation(self._current_path())

    def _observe(self, path=None):
        eta = np.array([
            self.state["position"][0], self.state["position"][1],
            self.state["position"][2], 0.0, self.state["pitch"],
            self.state["yaw"],
        ])
        sonar = self.sonar.observe(eta, self.obstacles)
        if self.sonar_history_length > 0:
            self._sonar_history.append(sonar.copy())
            frames = list(self._sonar_history)
            frames = [frames[0]] * (
                self._sonar_history.maxlen - len(frames)
            ) + frames
            frames = frames[::self.sonar_history_stride]
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
            float(np.clip(distance / 1000.0, 0.0, 1.0)),
        ], dtype=np.float32)
        motion = np.array([
            self.state["speed"],
            self.cruise_speed,
            self.speed_min,
            self.speed_max,
            self.state["yaw_rate"],
            self.state["pitch"],
            self.state["pitch_rate"],
        ], dtype=np.float32)
        observation = {
            "sonar": sonar,
            "target": target,
            "motion": motion,
        }
        if self.include_path_observation:
            path = self._current_path() if path is None else path
            observation["path"] = self._path_observation(path)
        if self.sonar_history_length > 0:
            observation["sonar_history"] = np.stack(frames)
        return observation

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
        self._sonar_history.clear()
        if self.use_dijkstra:
            for _attempt in range(10):
                self._sample_scene()
                self._build_cost_field()
                if np.isfinite(self._dijkstra_cost(self.state["position"])):
                    break
            else:
                raise RuntimeError("failed to generate a reachable obstacle scene")
            self._cached_path = self._plan_path(self.state["position"])
        else:
            self._sample_scene()
            self._cost_field = None
            self._cost_origin = None
            self._cost_occupancy = None
            self._cached_path = None
        self._initialize_obstacle_motion()
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
        if path is None or len(path) < 2:
            turn_factor = 1.0
        else:
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
        previous_position = self.state["position"].copy()
        old_distance = float(np.linalg.norm(self.goal - previous_position))
        self._update_obstacles()
        if (self.use_dijkstra
                and self.obstacle_motion == "random"
                and self.dijkstra_replan_steps > 0
                and self.steps > 0
                and self.steps % self.dijkstra_replan_steps == 0):
            self._build_cost_field()
            self._cached_path = None
        path = self._current_path() if self.use_dijkstra else None

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

        direct_progress = float(np.clip(old_distance - distance, -2.0, 2.0))
        if self.reward_mode == "dijkstra":
            cost_previous = self._dijkstra_cost(previous_position)
            cost_current = self._dijkstra_cost(position)
            if np.isfinite(cost_previous) and np.isfinite(cost_current):
                potential_previous = self._dijkstra_potential(previous_position)
                potential_current = self._dijkstra_potential(position)
                dijkstra_shaping = (
                    self.reward_gamma * potential_current - potential_previous
                )
                path_progress = (
                    dijkstra_shaping * self.dijkstra_potential_scale
                )
            else:
                path_progress = -1.0 if out_of_bounds else 0.0
                potential_current = float("-inf")
                dijkstra_shaping = 0.0
            distance_shaping = 0.0
            distance_potential = -distance / self.distance_potential_scale
        else:
            path_progress = 0.0
            potential_current = 0.0
            dijkstra_shaping = 0.0
            potential_previous = (
                -old_distance / self.distance_potential_scale
            )
            distance_potential = -distance / self.distance_potential_scale
            distance_shaping = (
                self.reward_gamma * distance_potential - potential_previous
            )
        observation = self._observe()
        alignment_error = (
            abs(float(self._path_observation(path)[1]))
            if path is not None and len(path) >= 2 else 0.0
        )
        desired_speed = self._desired_speed(path, distance)
        speed_error = abs(self.state["speed"] - desired_speed) / max(
            self.cruise_speed, 1.0
        )
        goal_proximity = 1.0 - float(np.clip(distance / 100.0, 0.0, 1.0))
        safety_penalty = max(0.0, 15.0 - clearance) / 15.0
        if self.reward_mode == "dijkstra":
            reward = (
                dijkstra_shaping
                + 2.0 * direct_progress
                - 0.5 * alignment_error
                - 0.3 * speed_error
                - 0.2 * safety_penalty
                - 0.02
                + 2.0 * goal_proximity
            )
        else:
            reward = (
                distance_shaping
                - 0.3 * speed_error
                - 0.2 * safety_penalty
                - 0.02
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
            "dijkstra_shaping": float(dijkstra_shaping),
            "dijkstra_potential": float(potential_current),
            "distance_shaping": float(distance_shaping),
            "distance_potential": float(distance_potential),
            "direct_progress": direct_progress,
            "alignment_error": alignment_error,
            "reward_mode": self.reward_mode,
            "obstacle_motion": self.obstacle_motion,
            "max_obstacle_speed_mps": float(self.obstacle_speeds.max())
                if self.obstacle_speeds.size else 0.0,
        }
        terminated = bool(collided or reached or out_of_bounds)
        truncated = bool(self.steps >= self.max_steps)
        if truncated and not terminated:
            reward -= 50.0
        return observation, float(reward), terminated, truncated, info

    def close(self):
        pass
