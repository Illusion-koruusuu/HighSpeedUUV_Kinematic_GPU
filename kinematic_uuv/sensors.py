"""Geometric three-view sonar proxy for the kinematic environment."""
import numpy as np
import torch


class SonarArray:
    def __init__(self, elevation_bins=12, azimuth_bins=48,
                 range_min=1.0, range_max=150.0):
        self.elevation_bins = int(elevation_bins)
        self.azimuth_bins = int(azimuth_bins)
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self._views = {
            "front": (0.0, 120.0, 30.0),
            "left": (-90.0, 120.0, 8.0),
            "right": (90.0, 120.0, 8.0),
        }
        self._angles = {
            name: (
                np.deg2rad(np.linspace(-elevation / 2.0, elevation / 2.0,
                                       self.elevation_bins)),
                np.deg2rad(np.linspace(yaw - azimuth / 2.0,
                                       yaw + azimuth / 2.0,
                                       self.azimuth_bins)),
            )
            for name, (yaw, azimuth, elevation) in self._views.items()
        }

    @property
    def observation_shape(self):
        return (6, self.elevation_bins, self.azimuth_bins)

    @staticmethod
    def _rotation_matrix(yaw, pitch, roll=0.0):
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        sy, cy = np.sin(yaw), np.cos(yaw)
        return np.array([
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ])

    def _view(self, eta, obstacles, view_name):
        elevation, azimuth = self._angles[view_name]
        elevation_grid, azimuth_grid = np.meshgrid(elevation, azimuth, indexing="ij")
        cos_elevation = np.cos(elevation_grid)
        local_directions = np.stack((
            cos_elevation * np.cos(azimuth_grid),
            cos_elevation * np.sin(azimuth_grid),
            np.sin(elevation_grid),
        ), axis=-1)
        rotation = self._rotation_matrix(eta[5], eta[4])
        directions = (local_directions @ rotation.T).reshape(-1, 3)
        nearest = np.full(len(directions), self.range_max, dtype=np.float32)
        origin = np.asarray(eta[:3], dtype=float)
        for center, radius in obstacles:
            offset = origin - np.asarray(center, dtype=float)
            b = directions @ offset
            c = float(np.dot(offset, offset) - radius * radius)
            discriminant = b * b - c
            valid = discriminant >= 0.0
            root = np.sqrt(np.maximum(discriminant, 0.0))
            near = -b - root
            far = -b + root
            distance = np.where(near >= 0.0, near, far)
            hits = valid & (distance >= self.range_min)
            nearest[hits] = np.minimum(nearest[hits], distance[hits])
        distances = nearest.reshape(self.elevation_bins, self.azimuth_bins)
        hit = (distances < self.range_max).astype(np.float32)
        normalized = np.clip(
            (distances - self.range_min) / (self.range_max - self.range_min),
            0.0, 1.0,
        )
        return normalized, hit

    def observe(self, eta, obstacles):
        channels = []
        for name in ("front", "left", "right"):
            distance, hit = self._view(eta, obstacles, name)
            channels.extend((distance, hit))
        return np.asarray(channels, dtype=np.float32)

    def near_obstacle(self, sonar, front_limit=45.0, side_limit=25.0):
        front_distance = sonar[0]
        front_hit = sonar[1] > 0.5
        central = slice(max(0, self.azimuth_bins // 2 - self.azimuth_bins // 6),
                        min(self.azimuth_bins,
                            self.azimuth_bins // 2 + self.azimuth_bins // 6))
        normalized_front = (front_limit - self.range_min) / (
            self.range_max - self.range_min
        )
        if np.any(front_hit[:, central] & (front_distance[:, central] <= normalized_front)):
            return True
        side_threshold = (side_limit - self.range_min) / (
            self.range_max - self.range_min
        )
        return bool(np.any((sonar[[3, 5]] > 0.5)
                           & (sonar[[2, 4]] <= side_threshold)))


class TorchSonarArray(SonarArray):
    """GPU-batched three-view sonar and segment collision checks.

    Obstacles are uploaded once per episode. Each simulation step only uploads
    the vehicle pose and downloads the small sonar tensor and collision flag.
    """

    def __init__(self, elevation_bins=12, azimuth_bins=48,
                 range_min=1.0, range_max=150.0, device="cuda"):
        super().__init__(elevation_bins, azimuth_bins, range_min, range_max)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"GPU sensor requested on {device}, but CUDA is unavailable"
            )
        self._local_directions = {}
        for name, (elevation, azimuth) in self._angles.items():
            elevation_grid, azimuth_grid = np.meshgrid(
                elevation, azimuth, indexing="ij"
            )
            cos_elevation = np.cos(elevation_grid)
            directions = np.stack((
                cos_elevation * np.cos(azimuth_grid),
                cos_elevation * np.sin(azimuth_grid),
                np.sin(elevation_grid),
            ), axis=-1).reshape(-1, 3)
            self._local_directions[name] = torch.as_tensor(
                directions, dtype=torch.float32, device=self.device
            )
        self._obstacle_source = None
        self._obstacle_centers = torch.empty(
            (0, 3), dtype=torch.float32, device=self.device
        )
        self._obstacle_radii = torch.empty(
            (0,), dtype=torch.float32, device=self.device
        )

    def set_obstacles(self, obstacles):
        if obstacles is self._obstacle_source:
            return
        if obstacles:
            centers = np.asarray([center for center, _ in obstacles], dtype=np.float32)
            radii = np.asarray([radius for _, radius in obstacles], dtype=np.float32)
            self._obstacle_centers = torch.as_tensor(
                centers, dtype=torch.float32, device=self.device
            )
            self._obstacle_radii = torch.as_tensor(
                radii, dtype=torch.float32, device=self.device
            )
        else:
            self._obstacle_centers = torch.empty(
                (0, 3), dtype=torch.float32, device=self.device
            )
            self._obstacle_radii = torch.empty(
                (0,), dtype=torch.float32, device=self.device
            )
        self._obstacle_source = obstacles

    @torch.inference_mode()
    def observe(self, eta, obstacles):
        self.set_obstacles(obstacles)
        roll, pitch, yaw = float(eta[3]), float(eta[4]), float(eta[5])
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        sy, cy = np.sin(yaw), np.cos(yaw)
        rotation = torch.as_tensor([
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ], dtype=torch.float32, device=self.device)
        origin = torch.as_tensor(
            eta[:3], dtype=torch.float32, device=self.device
        )
        channels = []
        if self._obstacle_centers.numel() == 0:
            for _ in range(3):
                channels.append(torch.ones(
                    (self.elevation_bins, self.azimuth_bins),
                    dtype=torch.float32, device=self.device,
                ))
                channels.append(torch.zeros_like(channels[-1]))
            return torch.stack(channels).cpu().numpy()

        offsets = origin.unsqueeze(0) - self._obstacle_centers
        radius_squared = self._obstacle_radii.square()
        for name in ("front", "left", "right"):
            directions = self._local_directions[name] @ rotation.T
            b = directions @ offsets.T
            c = offsets.square().sum(dim=1) - radius_squared
            discriminant = b.square() - c.unsqueeze(0)
            valid = discriminant >= 0.0
            root = torch.sqrt(torch.clamp(discriminant, min=0.0))
            near = -b - root
            far = -b + root
            distance = torch.where(near >= 0.0, near, far)
            distance = torch.where(
                valid & (distance >= self.range_min), distance,
                torch.full_like(distance, self.range_max),
            )
            nearest = distance.min(dim=1).values.reshape(
                self.elevation_bins, self.azimuth_bins
            )
            hit = (nearest < self.range_max).to(torch.float32)
            normalized = torch.clamp(
                (nearest - self.range_min) / (self.range_max - self.range_min),
                0.0, 1.0,
            )
            channels.extend((normalized, hit))
        return torch.stack(channels).cpu().numpy()

    @torch.inference_mode()
    def segment_hits_and_clearance(self, segment_start, segment_end,
                                   clearance_position=None):
        if self._obstacle_centers.numel() == 0:
            return False, 100.0
        start = torch.as_tensor(
            segment_start, dtype=torch.float32, device=self.device
        )
        end = torch.as_tensor(segment_end, dtype=torch.float32, device=self.device)
        segment = end - start
        length_squared = torch.dot(segment, segment)
        if float(length_squared) <= 1e-12:
            distances_squared = (
                self._obstacle_centers - start
            ).square().sum(dim=1)
        else:
            fraction = torch.clamp(
                ((self._obstacle_centers - start) @ segment) / length_squared,
                0.0, 1.0,
            )
            closest = start + fraction.unsqueeze(1) * segment
            distances_squared = (self._obstacle_centers - closest).square().sum(1)
        hit = bool((distances_squared <= self._obstacle_radii.square()).any())
        if clearance_position is None:
            return hit, 100.0
        point = torch.as_tensor(
            clearance_position, dtype=torch.float32, device=self.device
        )
        clearance = (
            (self._obstacle_centers - point).norm(dim=1)
            - self._obstacle_radii
        ).min()
        return hit, float(clearance)

    def segment_hits(self, segment_start, segment_end):
        return self.segment_hits_and_clearance(segment_start, segment_end)[0]
