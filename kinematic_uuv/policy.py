"""CNN feature extractor and bounded Gaussian PPO policy."""
import torch
from torch import nn
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SonarCNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        sonar_shape = observation_space.spaces["sonar"].shape
        self.sonar_cnn = nn.Sequential(
            nn.Conv2d(sonar_shape[0], 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sonar_features = self.sonar_cnn(
                torch.zeros(1, *sonar_shape)
            ).shape[1]
        self.state_keys = tuple(
            key for key in ("target", "path", "motion")
            if key in observation_space.spaces
        )
        state_dim = sum(observation_space.spaces[key].shape[0]
                        for key in self.state_keys)
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(sonar_features + 128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        sonar_features = self.sonar_cnn(observations["sonar"])
        state = torch.cat(tuple(observations[key] for key in self.state_keys), dim=1)
        return self.fusion(torch.cat(
            (sonar_features, self.state_mlp(state)), dim=1
        ))


class BoundedActorCriticPolicy(ActorCriticPolicy):
    """PPO policy whose Gaussian is squashed into the normalized action box."""

    def _build(self, lr_schedule):
        self.action_dist = SquashedDiagGaussianDistribution(
            get_action_dim(self.action_space)
        )
        super()._build(lr_schedule)
