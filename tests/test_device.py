import unittest
from pathlib import Path
import sys

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@unittest.skipIf(torch is None, "torch is not installed")
class GetDeviceTest(unittest.TestCase):
    def setUp(self):
        from learn_pytorch import get_device

        self.get_device = get_device

    def test_cpu_preference_returns_cpu(self):
        self.assertEqual(self.get_device("cpu"), torch.device("cpu"))

    def test_auto_returns_supported_device_type(self):
        self.assertIn(self.get_device().type, {"cpu", "cuda", "mps"})

    def test_unknown_preference_raises(self):
        with self.assertRaises(ValueError):
            self.get_device("tpu")


if __name__ == "__main__":
    unittest.main()
