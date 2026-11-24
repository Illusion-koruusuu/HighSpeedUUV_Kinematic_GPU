"""Procedural seabed height fields used by the kinematic environment."""
import numpy as np


class SeabedTerrain:
    """A static two-and-a-half-dimensional seabed.

    The solid seabed exists for every point whose vertical coordinate satisfies
    ``z <= height(x, y)``. The UUV therefore flies above the surface or around
    raised mountain ridges.
    """

    def __init__(self, height_grid, x_axis, y_axis):
        self.height_grid = np.asarray(height_grid, dtype=np.float32)
        self.x_axis = np.asarray(x_axis, dtype=np.float32)
        self.y_axis = np.asarray(y_axis, dtype=np.float32)
        if self.height_grid.shape != (len(self.y_axis), len(self.x_axis)):
            raise ValueError("terrain grid shape does not match axes")

    @classmethod
    def sample(cls, rng, planning_bounds, resolution=20.0,
               peak_count_range=(18, 30), base_height_range=(1.0, 7.0),
               max_height=62.0, start_xy=(0.0, 0.0)):
        lower = np.asarray(planning_bounds[0], dtype=float)
        upper = np.asarray(planning_bounds[1], dtype=float)
        x_axis = np.arange(lower[0], upper[0] + 0.5 * resolution, resolution)
        y_axis = np.arange(lower[1], upper[1] + 0.5 * resolution, resolution)
        grid_x, grid_y = np.meshgrid(x_axis, y_axis)

        base = float(rng.uniform(*base_height_range))
        height = (
            base
            + 1.8 * np.sin(grid_x / 145.0 + rng.uniform(0.0, 2.0 * np.pi))
            + 1.3 * np.cos(grid_y / 115.0 + rng.uniform(0.0, 2.0 * np.pi))
            + 0.7 * np.sin((grid_x + grid_y) / 61.0)
        )

        peak_count = int(rng.integers(
            int(peak_count_range[0]), int(peak_count_range[1]) + 1
        ))
        mountains = np.zeros_like(height)
        for _ in range(peak_count):
            center_x = rng.uniform(lower[0] + 40.0, upper[0] - 40.0)
            center_y = rng.uniform(lower[1] + 40.0, upper[1] - 40.0)
            sigma_major = rng.uniform(38.0, 135.0)
            sigma_minor = sigma_major * rng.uniform(0.35, 0.90)
            angle = rng.uniform(0.0, np.pi)
            dx = grid_x - center_x
            dy = grid_y - center_y
            along = dx * np.cos(angle) + dy * np.sin(angle)
            across = -dx * np.sin(angle) + dy * np.cos(angle)
            gaussian = np.exp(
                -0.5 * (
                    (along / sigma_major) ** 2
                    + (across / sigma_minor) ** 2
                )
            )
            amplitude = rng.uniform(0.20 * max_height, 0.92 * max_height)
            mountains += amplitude * gaussian

        roughness = (
            1.7 * np.sin(grid_x / 31.0 + grid_y / 47.0)
            + 1.1 * np.cos(grid_x / 23.0 - grid_y / 39.0)
        )
        height += mountains + roughness

        # Keep the starting cell open while preserving a continuous seabed.
        start_distance = np.hypot(
            grid_x - float(start_xy[0]), grid_y - float(start_xy[1])
        )
        start_cap = 2.5 + 0.06 * np.maximum(start_distance - 25.0, 0.0)
        height = np.minimum(height, start_cap)
        height = np.clip(height, 0.4, float(max_height))
        return cls(height, x_axis, y_axis)

    def height(self, x, y):
        """Return bilinearly interpolated terrain height."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x_fraction = np.clip(
            (x - self.x_axis[0]) / (self.x_axis[1] - self.x_axis[0]),
            0.0, len(self.x_axis) - 1.0,
        )
        y_fraction = np.clip(
            (y - self.y_axis[0]) / (self.y_axis[1] - self.y_axis[0]),
            0.0, len(self.y_axis) - 1.0,
        )
        x0 = np.floor(x_fraction).astype(np.intp)
        y0 = np.floor(y_fraction).astype(np.intp)
        x1 = np.minimum(x0 + 1, len(self.x_axis) - 1)
        y1 = np.minimum(y0 + 1, len(self.y_axis) - 1)
        fx = x_fraction - x0
        fy = y_fraction - y0
        value = (
            self.height_grid[y0, x0] * (1.0 - fx) * (1.0 - fy)
            + self.height_grid[y0, x1] * fx * (1.0 - fy)
            + self.height_grid[y1, x0] * (1.0 - fx) * fy
            + self.height_grid[y1, x1] * fx * fy
        )
        return value.astype(np.float32)

    def clearance(self, position):
        position = np.asarray(position, dtype=float)
        return float(position[..., 2] - self.height(
            position[..., 0], position[..., 1]
        ))

    def segment_clearance(self, start, end, max_spacing=1.0):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        length = float(np.linalg.norm(end - start))
        samples = max(2, int(np.ceil(length / max_spacing)) + 1)
        fractions = np.linspace(0.0, 1.0, samples)
        points = start[None, :] + fractions[:, None] * (end - start)[None, :]
        clearance = points[:, 2] - self.height(points[:, 0], points[:, 1])
        return bool(np.any(clearance <= 0.0)), float(np.min(clearance))

    def raycast(self, origin, directions, range_min=1.0,
                range_max=150.0, steps=64):
        """Return nearest terrain hit distance for each ray."""
        origin = np.asarray(origin, dtype=float)
        directions = np.asarray(directions, dtype=float)
        sample_distance = np.linspace(
            float(range_min), float(range_max), int(steps)
        )
        points = (
            origin[None, None, :]
            + sample_distance[None, :, None] * directions[:, None, :]
        )
        heights = self.height(points[..., 0], points[..., 1])
        below = (
            (points[..., 0] >= self.x_axis[0])
            & (points[..., 0] <= self.x_axis[-1])
            & (points[..., 1] >= self.y_axis[0])
            & (points[..., 1] <= self.y_axis[-1])
            & (points[..., 2] <= heights)
        )
        has_hit = below.any(axis=1)
        first = np.argmax(below, axis=1)
        distances = np.where(
            has_hit,
            sample_distance[first],
            float(range_max),
        )
        return distances.astype(np.float32)

    def metadata(self):
        return {
            "terrain_x": self.x_axis,
            "terrain_y": self.y_axis,
            "terrain_height": self.height_grid,
        }
