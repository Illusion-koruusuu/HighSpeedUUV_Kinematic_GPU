"""Three-dimensional Dijkstra cost field used for path observations."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def build_cost_field(goal, obstacles, start, resolution=8.0,
                     obstacle_margin=5.0, goal_radius=12.0,
                     planning_bounds=((-100.0, -350.0, 0.0),
                                      (700.0, 350.0, 100.0)),
                     vertical_cost_scale=2.0,
                     planner_goal_radius=0.0):
    """Build reverse distances to one goal cell or an optional goal region.

    The default targets only the grid cell nearest the goal center. A positive
    ``planner_goal_radius`` reproduces the former spherical goal-region mode.
    """
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    lower = np.asarray(planning_bounds[0], dtype=float)
    upper = np.asarray(planning_bounds[1], dtype=float)
    if np.any(upper <= lower):
        raise ValueError("invalid planning bounds")
    lower = np.floor(lower / resolution) * resolution
    upper = np.ceil(upper / resolution) * resolution
    shape = tuple(np.floor((upper - lower) / resolution).astype(int) + 1)
    axes = [lower[i] + np.arange(shape[i]) * resolution for i in range(3)]
    grid_x, grid_y, grid_z = np.meshgrid(*axes, indexing="ij")

    occupancy = np.zeros(shape, dtype=bool)
    for center, radius in obstacles:
        center = np.asarray(center, dtype=float)
        inflated = float(radius) + obstacle_margin
        occupancy |= ((grid_x - center[0]) ** 2
                      + (grid_y - center[1]) ** 2
                      + (grid_z - center[2]) ** 2 <= inflated ** 2)

    if planner_goal_radius > 0.0:
        goal_grid = ((grid_x - goal[0]) ** 2
                     + (grid_y - goal[1]) ** 2
                     + (grid_z - goal[2]) ** 2 <= planner_goal_radius ** 2)
        goal_grid &= ~occupancy
    else:
        goal_grid = np.zeros(shape, dtype=bool)
    if not goal_grid.any():
        index = tuple(np.clip(
            np.rint((goal - lower) / resolution).astype(int),
            0, np.asarray(shape) - 1,
        ))
        goal_grid[index] = True
        occupancy[index] = False

    free_mask = ~occupancy
    free_indices = np.flatnonzero(free_mask)
    node_map = np.full(occupancy.size, -1, dtype=np.int32)
    node_map[free_indices] = np.arange(free_indices.size, dtype=np.int32)
    coordinates = np.stack(np.unravel_index(free_indices, shape), axis=1)
    strides = np.array([
        shape[1] * shape[2],
        shape[2],
        1,
    ], dtype=np.int64)
    flat_indices = coordinates @ strides
    shape_array = np.asarray(shape)

    edge_sources = []
    edge_targets = []
    edge_weights = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                length = np.linalg.norm((dx, dy, vertical_cost_scale * dz))
                neighbor_coordinates = coordinates + (dx, dy, dz)
                inside = np.all(
                    (neighbor_coordinates >= 0)
                    & (neighbor_coordinates < shape_array),
                    axis=1,
                )
                neighbor_flat = flat_indices + int(
                    np.dot((dx, dy, dz), strides)
                )
                neighbor_nodes = np.full(free_indices.size, -1, dtype=np.int32)
                neighbor_nodes[inside] = node_map[neighbor_flat[inside]]
                connected = neighbor_nodes >= 0
                sources = np.flatnonzero(connected)
                edge_sources.append(sources)
                edge_targets.append(neighbor_nodes[connected])
                edge_weights.append(np.full(
                    sources.size, resolution * float(length), dtype=np.float32
                ))

    graph = csr_matrix(
        (
            np.concatenate(edge_weights),
            (np.concatenate(edge_sources), np.concatenate(edge_targets)),
        ),
        shape=(free_indices.size, free_indices.size),
    )
    goal_nodes = node_map[np.flatnonzero(goal_grid)]
    goal_nodes = goal_nodes[goal_nodes >= 0]
    costs = np.full(shape, np.inf, dtype=np.float32)
    if goal_nodes.size:
        distances = dijkstra(
            graph,
            directed=False,
            indices=goal_nodes,
            min_only=True,
        )
        costs.ravel()[free_indices] = distances.astype(np.float32)
    return costs, lower, float(resolution), occupancy


def interpolate_cost(costs, origin, resolution, position):
    coordinate = (np.asarray(position, dtype=float) - origin) / resolution
    if np.any(coordinate < 0.0) or np.any(coordinate > np.asarray(costs.shape) - 1):
        return float("inf")
    base = np.floor(coordinate).astype(int)
    fraction = coordinate - base
    value = 0.0
    weight_sum = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                index = base + np.array([dx, dy, dz])
                if np.any(index >= np.asarray(costs.shape)):
                    continue
                weight = float(np.prod(np.where(
                    np.array([dx, dy, dz], dtype=bool),
                    fraction, 1.0 - fraction,
                )))
                cost = float(costs[tuple(index)])
                if np.isfinite(cost):
                    value += weight * cost
                    weight_sum += weight
    return value / weight_sum if weight_sum > 0.0 else float("inf")
