import unittest

import numpy as np

from training.teacher_data import collect_teacher_episode


class TeacherDataTests(unittest.TestCase):
    def test_successful_teacher_episode_is_exported(self):
        episode = collect_teacher_episode((
            40.0,
            5,
            800,
            {
                "obstacle_count_range": (20, 20),
                "obstacle_motion": "static",
                "use_dijkstra": True,
                "include_path_observation": False,
                "reward_mode": "euclidean",
                "sonar_history_length": 8,
                "sonar_history_stride": 4,
            },
        ))
        self.assertIsNotNone(episode)
        self.assertGreater(len(episode["actions"]), 0)
        self.assertEqual(episode["actions"].shape[1], 3)
        self.assertEqual(
            episode["observations"]["sonar_history"].shape[2:],
            (6, 12, 48),
        )
        self.assertNotIn("sonar", episode["observations"])
        self.assertNotIn("path", episode["observations"])
        self.assertEqual(
            episode["observations"]["sonar_history"].dtype, np.float16
        )


if __name__ == "__main__":
    unittest.main()
