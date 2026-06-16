import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@unittest.skipIf(torch is None, "torch is not installed")
class AutogradHelpersTest(unittest.TestCase):
    def setUp(self):
        from learn_pytorch import (
            make_linear_regression_data,
            mean_squared_error,
            predict_linear,
            train_manual_linear_regression,
        )

        self.make_linear_regression_data = make_linear_regression_data
        self.mean_squared_error = mean_squared_error
        self.predict_linear = predict_linear
        self.train_manual_linear_regression = train_manual_linear_regression

    def test_make_linear_regression_data_shapes(self):
        data = self.make_linear_regression_data(num_samples=8, noise_std=0.0)

        self.assertEqual(data.x.shape, torch.Size([8, 1]))
        self.assertEqual(data.y.shape, torch.Size([8, 1]))
        self.assertEqual(data.x.dtype, torch.float32)
        self.assertEqual(data.y.dtype, torch.float32)

    def test_mean_squared_error_matches_manual_formula(self):
        prediction = torch.tensor([[1.0], [3.0]])
        target = torch.tensor([[2.0], [1.0]])

        loss = self.mean_squared_error(prediction, target)

        self.assertAlmostEqual(float(loss), 2.5)

    def test_manual_linear_regression_converges_without_noise(self):
        data = self.make_linear_regression_data(
            num_samples=64,
            weight=3.0,
            bias=-0.5,
            noise_std=0.0,
            seed=7,
        )

        result = self.train_manual_linear_regression(
            data.x,
            data.y,
            learning_rate=0.2,
            epochs=300,
            seed=0,
        )

        self.assertLess(result.losses[-1], result.losses[0])
        self.assertAlmostEqual(result.weight, data.true_weight, delta=0.05)
        self.assertAlmostEqual(result.bias, data.true_bias, delta=0.05)

    def test_mean_squared_error_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            self.mean_squared_error(torch.zeros(2, 1), torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
