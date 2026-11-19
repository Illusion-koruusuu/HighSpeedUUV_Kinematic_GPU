import unittest

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

from kinematic_uuv.env import KinematicUUVEnv
from kinematic_uuv.policy import BoundedActorCriticPolicy, SonarCNNFeaturesExtractor


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


if __name__ == "__main__":
    unittest.main()
