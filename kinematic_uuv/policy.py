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
        history_space = observation_space.spaces.get("sonar_history")
        sonar_shape = (
            history_space.shape[1:]
            if history_space is not None
            else observation_space.spaces["sonar"].shape
        )
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
        self.temporal_lstm = None
        if history_space is not None:
            self.temporal_lstm = nn.LSTM(
                input_size=sonar_features,
                hidden_size=128,
                batch_first=True,
            )
            sonar_features = 128
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
        if self.temporal_lstm is None:
            sonar_features = self.sonar_cnn(observations["sonar"])
        else:
            history = observations["sonar_history"]
            batch_size, history_length = history.shape[:2]
            frames = history.reshape(
                batch_size * history_length, *history.shape[2:]
            )
            frame_features = self.sonar_cnn(frames).reshape(
                batch_size, history_length, -1
            )
            _, (hidden_state, _) = self.temporal_lstm(frame_features)
            sonar_features = hidden_state[-1]
        state = torch.cat(tuple(observations[key] for key in self.state_keys), dim=1)
        return self.fusion(torch.cat(
            (sonar_features, self.state_mlp(state)), dim=1
        ))


class BoundedActorCriticPolicy(ActorCriticPolicy):
    """PPO policy whose Gaussian is squashed into the normalized action box."""

    def __init__(self, *args, residual_scale=None, **kwargs):
        self.residual_scale = (
            None if residual_scale is None else float(residual_scale)
        )
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        self.action_dist = SquashedDiagGaussianDistribution(
            get_action_dim(self.action_space)
        )
        super()._build(lr_schedule)
        self.residual_action_net = None
        if self.residual_scale is not None:
            self.residual_action_net = nn.Linear(
                self.mlp_extractor.latent_dim_pi,
                get_action_dim(self.action_space),
            )
            nn.init.zeros_(self.residual_action_net.weight)
            nn.init.zeros_(self.residual_action_net.bias)
            self.optimizer.add_param_group({
                "params": self.residual_action_net.parameters(),
                "lr": self.optimizer.param_groups[0]["lr"],
            })
            self.residual_action_net.requires_grad_(False)

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        if self.residual_action_net is not None:
            mean_actions = mean_actions + (
                self.residual_scale
                * self.residual_action_net(latent_pi)
            )
        return self.action_dist.proba_distribution(
            mean_actions, self.log_std
        )

    def set_bc_mode(self, enabled):
        if self.residual_action_net is not None:
            self.residual_action_net.requires_grad_(False)
        self.features_extractor.requires_grad_(True)
        self.action_net.requires_grad_(True)
        self.value_net.requires_grad_(True)

    def prepare_residual_ppo(self):
        if self.residual_action_net is None:
            return
        self.features_extractor.requires_grad_(False)
        self.action_net.requires_grad_(False)
        self.log_std.requires_grad_(False)
        self.residual_action_net.requires_grad_(False)
        self.value_net.requires_grad_(True)

    def enable_residual_updates(self):
        if self.residual_action_net is not None:
            self.residual_action_net.requires_grad_(True)
            self.log_std.requires_grad_(True)
