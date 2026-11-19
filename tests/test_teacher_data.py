import unittest

from training.teacher_data import collect_teacher_episode


class TeacherDataTests(unittest.TestCase):
    def test_successful_teacher_episode_is_exported(self):
        episode = collect_teacher_episode((40.0, 5, 800))
        self.assertIsNotNone(episode)
        self.assertGreater(len(episode["actions"]), 0)
        self.assertEqual(episode["actions"].shape[1], 3)
        self.assertEqual(
            episode["observations"]["sonar"].shape[1:],
            (6, 12, 48),
        )


if __name__ == "__main__":
    unittest.main()
