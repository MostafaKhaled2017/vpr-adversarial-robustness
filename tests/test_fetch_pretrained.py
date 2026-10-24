import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import dataclasses

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "third_party" / "SuperVLAD"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src import fetch_pretrained  # noqa: E402
from src.models import get_model_adapter  # noqa: E402


def perturb(model, args):
    """Stand-in for download_weights: weights that differ from a freshly built model."""
    del args
    with torch.no_grad():
        model.aggregator.row_proj.weight.fill_(0.5)


class FetchPretrainedTests(unittest.TestCase):
    def test_saved_checkpoint_loads_back_strictly(self):
        adapter = dataclasses.replace(get_model_adapter("dinomix"), download_weights=perturb)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(fetch_pretrained, "get_model_adapter", return_value=adapter):
            out = Path(tmp) / "base.pth"
            self.assertEqual(fetch_pretrained.main(["--model", "dinomix", "--out", str(out)]), 0)
            args = types.SimpleNamespace(freeze_te=None, device="cpu")
            model = adapter.build(args).model
            adapter.load_weights(model, str(out), args)
        self.assertTrue(torch.all(model.aggregator.row_proj.weight == 0.5))

    def test_existing_out_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "base.pth"
            out.write_bytes(b"keep")
            with mock.patch.object(fetch_pretrained, "get_model_adapter") as get_adapter:
                self.assertEqual(fetch_pretrained.main(["--model", "qaa", "--out", str(out)]), 1)
            get_adapter.assert_not_called()
            self.assertEqual(out.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
