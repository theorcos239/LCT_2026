"""Проверка выбора GPU без необходимости иметь его в тестовой среде."""

import unittest
from unittest.mock import patch

import torch

from .device import select_device


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_selects_mps_on_apple_gpu(self):
        with (patch.object(torch.cuda, "is_available", return_value=False),
              patch.object(torch.backends.mps, "is_available", return_value=True)):
            self.assertEqual(select_device(), torch.device("mps"))

    def test_auto_selects_xpu_when_only_intel_gpu_is_available(self):
        with (patch.object(torch.cuda, "is_available", return_value=False),
              patch.object(torch.backends.mps, "is_available", return_value=False),
              patch.object(torch.xpu, "is_available", return_value=True)):
            self.assertEqual(select_device(), torch.device("xpu"))

    def test_explicit_unavailable_gpu_fails(self):
        with patch.object(torch.backends.mps, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "mps.*недоступно"):
                select_device("mps")


if __name__ == "__main__":
    unittest.main()
