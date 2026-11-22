import unittest

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

from kinematic_uuv.env import KinematicUUVEnv
from kinematic_uuv.policy import BoundedActorCriticPolicy, SonarCNNFeaturesExtractor
from training.train import _policy_action_mse


class PolicyTests(unittest.TestCase):
    def test_bounded_policy_outputs_valid_actions(self):
        env = KinematicUUVEnv(max_steps=4)
        try:
            model = PPO(
                BoundedActorCriticPolicy, env, n_steps=8, batch_size=8,
                policy_kwargs={"features_extractor_class": SonarCNNFeaturesExtractor},
            )
            self.assertIsInstance(
                model.policy.action_dist, SquashedDiagGaussianDistribution
            )
            observation, _ = env.reset(seed=0)
            action, _ = model.predict(observation, deterministic=False)
            self.assertTrue(np.all(action >= -1.0))
            self.assertTrue(np.all(action <= 1.0))
        finally:
            env.close()

    def test_lstm_and_residual_policy_shapes(self):
        env = KinematicUUVEnv(
            max_steps=4,
            obstacle_count_range=(20, 20),
            sonar_history_length=3,
            use_dijkstra=False,
            include_path_observation=False,
            reward_mode="euclidean",
        )
        try:
            model = PPO(
                BoundedActorCriticPolicy, env, n_steps=8, batch_size=8,
                policy_kwargs={
                    "features_extractor_class": SonarCNNFeaturesExtractor,
                    "residual_scale": 0.15,
                },
            )
            self.assertIsNotNone(
                model.policy.features_extractor.temporal_lstm
            )
            self.assertIsNotNone(model.policy.residual_action_net)
            observation, _ = env.reset(seed=0)
            self.assertNotIn("path", observation)
            action, _ = model.predict(observation, deterministic=True)
            self.assertEqual(action.shape, (3,))
            model.policy.prepare_residual_ppo()
            self.assertFalse(
                model.policy.residual_action_net.weight.requires_grad
            )
            self.assertFalse(model.policy.log_std.requires_grad)
            model.policy.enable_residual_updates()
            self.assertTrue(
                model.policy.residual_action_net.weight.requires_grad
            )
            self.assertTrue(model.policy.log_std.requires_grad)
        finally:
            env.close()

    def test_policy_action_mse_restores_training_mode(self):
        env = KinematicUUVEnv(
            max_steps=4,
            obstacle_count_range=(20, 20),
            sonar_history_length=3,
        )
        try:
            model = PPO(
                BoundedActorCriticPolicy, env, n_steps=8, batch_size=8,
                policy_kwargs={
                    "features_extractor_class": SonarCNNFeaturesExtractor,
                },
            )
            observation, _ = env.reset(seed=0)
            dataset = {
                "observations": {
                    key: np.asarray([value], dtype=np.float32)
                    for key, value in observation.items()
                },
                "actions": np.zeros((1, 3), dtype=np.float32),
            }

            model.policy.set_training_mode(True)
            _policy_action_mse(model, dataset, batch_size=1)
            self.assertTrue(model.policy.training)
            self.assertTrue(model.policy.features_extractor.temporal_lstm.training)

            model.policy.set_training_mode(False)
            _policy_action_mse(model, dataset, batch_size=1)
            self.assertFalse(model.policy.training)
            self.assertFalse(model.policy.features_extractor.temporal_lstm.training)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
