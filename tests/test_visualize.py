import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.visualize import _summary, _trajectory_files, plot_trajectory


class VisualizationTests(unittest.TestCase):
    def test_trajectory_files_can_be_filtered_by_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for speed in (10, 20):
                for seed in (0, 5):
                    (root / f"speed{speed}_seed{seed}.npz").touch()
            files = _trajectory_files(
                root, speeds=[20.0], seeds=[5], max_trajectories=10
            )
            self.assertEqual(
                [file.name for file in files],
                ["speed20_seed5.npz"],
            )

    def test_trajectory_diagnostics_are_created(self):
        steps = 20
        position = np.column_stack((
            np.linspace(0.0, 40.0, steps + 1),
            np.zeros(steps + 1),
            np.linspace(10.0, 8.0, steps + 1),
        ))
        data = {
            "position": position,
            "roll_pitch_yaw": np.zeros((steps + 1, 3)),
            "speed": np.full(steps + 1, 5.0),
            "yaw_rate": np.zeros(steps + 1),
            "goal": np.array([40.0, 0.0, 8.0]),
            "goal_radius_m": np.asarray(5.0),
            "obstacles": np.array([[20.0, 8.0, 8.0, 3.0]]),
            "success": np.asarray(True),
            "target_speed_knots": np.asarray(10.0),
            "seed": np.asarray(7),
        }
        summary = _summary(data, 0.1)
        self.assertTrue(summary["success"])
        self.assertGreater(summary["path_length_m"], 40.0)
        self.assertAlmostEqual(summary["final_distance_to_goal_m"], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostics.png"
            plot_trajectory(data, output, 0.1)
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
