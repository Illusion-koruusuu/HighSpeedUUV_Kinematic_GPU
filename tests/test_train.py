import unittest

import numpy as np

from training.train import _merge_datasets


class TrainDataTests(unittest.TestCase):
    def test_merged_dataset_indexes_without_copying_arrays(self):
        first = {
            "observations": {
                "value": np.arange(6, dtype=np.float32).reshape(2, 3),
            },
            "actions": np.zeros((2, 3), dtype=np.float32),
            "steps": 2,
        }
        second = {
            "observations": {
                "value": np.arange(9, dtype=np.float32).reshape(3, 3),
            },
            "actions": np.ones((3, 3), dtype=np.float32),
            "steps": 3,
        }
        merged = _merge_datasets(first, second)
        self.assertEqual(merged["steps"], 5)
        np.testing.assert_array_equal(
            merged["observations"]["value"][np.array([0, 2, 4])],
            np.array([[0, 1, 2], [0, 1, 2], [6, 7, 8]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            merged["actions"][2:],
            np.ones((3, 3), dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
