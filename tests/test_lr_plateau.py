import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.checkpoints import should_drop_lr_on_plateau


class LrPlateauTests(unittest.TestCase):
    def test_disabled_when_patience_is_none(self):
        self.assertFalse(should_drop_lr_on_plateau(5, None))

    def test_never_drops_when_improving(self):
        self.assertFalse(should_drop_lr_on_plateau(0, 5))

    def test_drops_exactly_on_patience_multiples(self):
        self.assertFalse(should_drop_lr_on_plateau(4, 5))
        self.assertTrue(should_drop_lr_on_plateau(5, 5))
        self.assertFalse(should_drop_lr_on_plateau(6, 5))
        self.assertTrue(should_drop_lr_on_plateau(10, 5))


if __name__ == "__main__":
    unittest.main()
