import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import resolve_mode


class ResolveModeTests(unittest.TestCase):
    def test_defaults_to_rank(self):
        mode, forwarded = resolve_mode(["--model_type=supervlad", "--epsilons", "0.01"])
        self.assertEqual(mode, "rank")
        self.assertEqual(forwarded, ["--model_type=supervlad", "--epsilons", "0.01"])

    def test_empty_argv_defaults_to_rank(self):
        self.assertEqual(resolve_mode([]), ("rank", []))

    def test_explicit_rank_token_is_consumed(self):
        mode, forwarded = resolve_mode(["rank", "--epsilons", "0.01"])
        self.assertEqual(mode, "rank")
        self.assertEqual(forwarded, ["--epsilons", "0.01"])

    def test_perceptual_token_selects_perceptual(self):
        mode, forwarded = resolve_mode(["perceptual", "--datasets", "sped"])
        self.assertEqual(mode, "perceptual")
        self.assertEqual(forwarded, ["--datasets", "sped"])

    def test_mode_token_only_recognized_in_first_position(self):
        mode, forwarded = resolve_mode(["--model_tags", "perceptual"])
        self.assertEqual(mode, "rank")
        self.assertEqual(forwarded, ["--model_tags", "perceptual"])

    def test_importing_eval_module_stays_lightweight(self):
        self.assertNotIn("torch", sys.modules.get("eval").__dict__)


if __name__ == "__main__":
    unittest.main()
