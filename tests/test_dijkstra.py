import unittest

import numpy as np

from kinematic_uuv.dijkstra import build_cost_field, interpolate_cost


class DijkstraTests(unittest.TestCase):
    def test_goal_region_has_zero_cost(self):
        goal = np.array([100.0, 0.0, 10.0])
        costs, origin, resolution, occupancy = build_cost_field(
            goal, [], start=np.zeros(3), resolution=10.0,
            planning_bounds=((-50.0, -50.0, 0.0), (200.0, 50.0, 50.0)),
        )
        self.assertTrue(np.isfinite(costs).any())
        self.assertLessEqual(interpolate_cost(costs, origin, resolution, goal), 1e-5)
        self.assertFalse(occupancy.all())

    def test_planner_can_target_the_goal_center_only(self):
        goal = np.array([100.0, 0.0, 10.0])
        costs, origin, resolution, occupancy = build_cost_field(
            goal, [], start=np.zeros(3), resolution=10.0,
            goal_radius=5.0,
            planning_bounds=((-50.0, -50.0, 0.0), (200.0, 50.0, 50.0)),
            planner_goal_radius=0.0,
        )
        goal_index = tuple(np.rint(
            (goal - origin) / resolution
        ).astype(int))
        self.assertEqual(int(np.count_nonzero(costs <= 1e-5)), 1)
        self.assertEqual(float(costs[goal_index]), 0.0)
        self.assertFalse(occupancy[goal_index])


if __name__ == "__main__":
    unittest.main()
