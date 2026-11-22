import unittest

import numpy as np

from kinematic_uuv.env import KNOT_MPS, KinematicUUVEnv
from kinematic_uuv.teacher import teacher_action


class KinematicEnvTests(unittest.TestCase):
    def test_default_goal_distance_and_bounds(self):
        env = KinematicUUVEnv(obstacle_count_range=(0, 0))
        try:
            for seed in range(1000):
                env.np_random = np.random.default_rng(seed)
                env._sample_scene()
                distance = np.linalg.norm(env.goal - [0.0, 0.0, 10.0])
                self.assertGreaterEqual(distance, 650.0)
                self.assertLessEqual(distance, 900.0)
                self.assertFalse(env._outside_bounds(env.goal))
        finally:
            env.close()

    def test_reset_observation_and_scene(self):
        env = KinematicUUVEnv(max_steps=20)
        observation, _ = env.reset(seed=0)
        try:
            self.assertEqual(observation["sonar"].shape, (6, 12, 48))
            self.assertEqual(observation["target"].shape, (4,))
            self.assertEqual(observation["path"].shape, (5,))
            self.assertEqual(observation["motion"].shape, (7,))
            self.assertLess(observation["target"][3], 1.0)
            for value in observation.values():
                self.assertTrue(np.isfinite(value).all())
            self.assertGreaterEqual(len(env.obstacles), env.obstacle_count_min)
            self.assertLessEqual(len(env.obstacles), env.obstacle_count_max)
        finally:
            env.close()

    def test_kinematic_limits_are_enforced(self):
        env = KinematicUUVEnv(max_steps=200, min_turn_radius=30.0,
                              max_lat_accel=4.0)
        observation, _ = env.reset(seed=1)
        env.cruise_speed_override = 50.0 * KNOT_MPS
        observation, _ = env.reset(seed=1)
        max_lat_accel = 0.0
        try:
            for _ in range(100):
                action = teacher_action(observation)
                observation, _, terminated, truncated, info = env.step(action)
                max_lat_accel = max(
                    max_lat_accel,
                    env.state["speed"] * abs(env.state["yaw_rate"]),
                )
                self.assertLessEqual(env.state["speed"], env.speed_max + 1e-9)
                self.assertLessEqual(abs(env.state["pitch"]), env.max_pitch + 1e-9)
                self.assertLessEqual(max_lat_accel, env.max_lat_accel + 0.05)
                if terminated or truncated:
                    break
        finally:
            env.close()

    def test_default_scene_has_uniform_dense_obstacles(self):
        env = KinematicUUVEnv(max_steps=20)
        try:
            env.reset(seed=0)
            obstacles = env.obstacles
            start = env.state["position"]
            centers = np.asarray([center for center, _ in obstacles])
            self.assertGreaterEqual(len(obstacles), 20)
            self.assertTrue(any(radius >= 14.0 for _, radius in obstacles))
            self.assertGreater(np.ptp(centers[:, 0]), 250.0)
            self.assertGreater(np.ptp(centers[:, 1]), 250.0)
            self.assertGreater(np.ptp(centers[:, 2]), 30.0)
            self.assertTrue(np.isfinite(env._dijkstra_cost(start)))
        finally:
            env.close()

    def test_dijkstra_reward_uses_discounted_potential(self):
        env = KinematicUUVEnv(max_steps=20)
        observation, _ = env.reset(seed=0)
        previous = env.state["position"].copy()
        try:
            observation, _, _, _, info = env.step(
                teacher_action(observation)
            )
            current = env.state["position"].copy()
            expected = (
                env.reward_gamma * env._dijkstra_potential(current)
                - env._dijkstra_potential(previous)
            )
            self.assertAlmostEqual(
                info["dijkstra_shaping"], expected, places=6
            )
        finally:
            env.close()

    def test_default_obstacles_are_static(self):
        env = KinematicUUVEnv(max_steps=20, obstacle_count_range=(20, 20))
        observation, _ = env.reset(seed=0)
        before = np.asarray([center for center, _ in env.obstacles])
        try:
            observation, _, _, _, _ = env.step(
                teacher_action(observation)
            )
            after = np.asarray([center for center, _ in env.obstacles])
            np.testing.assert_allclose(after, before)
        finally:
            env.close()

    def test_random_obstacles_move_within_uuv_speed_limit(self):
        env = KinematicUUVEnv(
            max_steps=20,
            obstacle_count_range=(20, 20),
            obstacle_motion="random",
            obstacle_speed_range=(KNOT_MPS, 10.0 * KNOT_MPS),
        )
        observation, _ = env.reset(seed=0)
        before = np.asarray([center for center, _ in env.obstacles])
        try:
            observation, _, _, _, info = env.step(
                teacher_action(observation)
            )
            after = np.asarray([center for center, _ in env.obstacles])
            self.assertTrue(np.any(np.linalg.norm(after - before, axis=1) > 0.0))
            self.assertLessEqual(
                info["max_obstacle_speed_mps"], 10.0 * KNOT_MPS
            )
            self.assertEqual(info["obstacle_motion"], "random")
        finally:
            env.close()

    def test_dynamic_obstacles_rebuild_dijkstra_periodically(self):
        env = KinematicUUVEnv(
            max_steps=20,
            obstacle_count_range=(20, 20),
            obstacle_motion="random",
            dijkstra_replan_steps=2,
        )
        observation, _ = env.reset(seed=0)
        initial_field = env._cost_field
        try:
            for _ in range(3):
                observation, _, terminated, truncated, _ = env.step(
                    teacher_action(observation)
                )
                if terminated or truncated:
                    break
            self.assertIsNot(env._cost_field, initial_field)
        finally:
            env.close()

    def test_sonar_history_is_added_when_requested(self):
        env = KinematicUUVEnv(
            max_steps=20,
            obstacle_count_range=(20, 20),
            sonar_history_length=3,
        )
        try:
            observation, _ = env.reset(seed=0)
            self.assertEqual(observation["sonar_history"].shape, (3, 6, 12, 48))
            np.testing.assert_allclose(
                observation["sonar_history"][-1], observation["sonar"]
            )
        finally:
            env.close()

    def test_sonar_history_supports_temporal_stride(self):
        env = KinematicUUVEnv(
            max_steps=20,
            obstacle_count_range=(20, 20),
            sonar_history_length=3,
            sonar_history_stride=4,
            use_dijkstra=False,
            include_path_observation=False,
            reward_mode="euclidean",
        )
        try:
            observation, _ = env.reset(seed=0)
            sonar_sequence = [observation["sonar"].copy()]
            observation = None
            for _ in range(9):
                observation, _, terminated, truncated, _ = env.step(
                    np.zeros(3, dtype=np.float32)
                )
                sonar_sequence.append(observation["sonar"].copy())
                if terminated or truncated:
                    break
            np.testing.assert_allclose(
                observation["sonar_history"][-1], sonar_sequence[-1]
            )
            np.testing.assert_allclose(
                observation["sonar_history"][-2], sonar_sequence[-5]
            )
            np.testing.assert_allclose(
                observation["sonar_history"][-3], sonar_sequence[-9]
            )
            self.assertEqual(env._sonar_history.maxlen, 9)
        finally:
            env.close()

    def test_path_free_euclidean_mode_skips_dijkstra(self):
        env = KinematicUUVEnv(
            max_steps=20,
            obstacle_count_range=(20, 20),
            sonar_history_length=3,
            use_dijkstra=False,
            include_path_observation=False,
            reward_mode="euclidean",
            distance_potential_scale=250.0,
        )
        try:
            observation, _ = env.reset(seed=0)
            self.assertNotIn("path", observation)
            self.assertIsNone(env._cost_field)
            previous_distance = float(np.linalg.norm(
                env.goal - env.state["position"]
            ))
            observation, _, terminated, truncated, info = env.step(
                np.zeros(3, dtype=np.float32)
            )
            current_distance = float(np.linalg.norm(
                env.goal - env.state["position"]
            ))
            expected = (
                env.reward_gamma * (-current_distance / 250.0)
                - (-previous_distance / 250.0)
            )
            self.assertAlmostEqual(info["distance_shaping"], expected, places=7)
            self.assertEqual(info["reward_mode"], "euclidean")
            self.assertEqual(info["dijkstra_shaping"], 0.0)
            self.assertIsNone(env._cost_field)
        finally:
            env.close()

    def test_teacher_path_is_available_without_student_path_observation(self):
        env = KinematicUUVEnv(
            max_steps=20,
            obstacle_count_range=(20, 20),
            use_dijkstra=True,
            include_path_observation=False,
            reward_mode="euclidean",
        )
        try:
            observation, _ = env.reset(seed=0)
            self.assertNotIn("path", observation)
            self.assertIsNotNone(env._cost_field)
            self.assertEqual(env.teacher_path_observation().shape, (5,))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
