import unittest

import numpy as np
import torch

from kinematic_uuv.env import KinematicUUVEnv


class GpuSensorTests(unittest.TestCase):
    def _compare(self, sensor_device):
        common = dict(max_steps=5, obstacle_count_range=(20, 20))
        cpu_env = KinematicUUVEnv(sensor_backend="cpu", **common)
        gpu_env = KinematicUUVEnv(
            sensor_backend="gpu", sensor_device=sensor_device, **common
        )
        try:
            cpu_observation, _ = cpu_env.reset(seed=123)
            gpu_observation, _ = gpu_env.reset(seed=123)
            np.testing.assert_allclose(
                cpu_observation["sonar"], gpu_observation["sonar"],
                rtol=1e-4, atol=1e-4,
            )
            start = cpu_env.state["position"].copy()
            end = start + np.array([10.0, 2.0, -1.0])
            cpu_hit = any(
                cpu_env._segment_distance(center, start, end) <= radius
                for center, radius in cpu_env.obstacles
            )
            gpu_hit, _ = gpu_env.sonar.segment_hits_and_clearance(
                start, end, end
            )
            self.assertEqual(cpu_hit, gpu_hit)
        finally:
            cpu_env.close()
            gpu_env.close()

    def test_torch_sensor_matches_numpy_on_cpu(self):
        self._compare("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_sensor_matches_numpy(self):
        self._compare("cuda")


if __name__ == "__main__":
    unittest.main()
