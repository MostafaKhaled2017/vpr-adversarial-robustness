import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load the root dispatcher by file path rather than by module name. Importing
# it as plain `eval` is unsafe here: third_party/SuperVLAD/eval.py sits ahead
# of the repo root on sys.path once any test that triggers a SuperVLAD
# bootstrap (e.g. tests.test_faiss_utils) has already run, so `import eval`
# can resolve to SuperVLAD's eval.py instead of the root dispatcher and crash
# at import time (it calls parser.parse_arguments() at module scope).
_spec = importlib.util.spec_from_file_location("root_eval", REPO_ROOT / "eval.py")
root_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(root_eval)

resolve_mode = root_eval.resolve_mode


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
        self.assertNotIn("torch", root_eval.__dict__)

    def test_importing_eval_module_stays_lightweight_in_subprocess(self):
        # Belt-and-suspenders check run in a fresh interpreter: importing the
        # root dispatcher as `eval` (its real import name for downstream
        # users) must not pull in torch/numpy or any src.* module, and must
        # not accidentally resolve to third_party/SuperVLAD/eval.py.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import eval, sys; "
                "assert 'torch' not in sys.modules and 'numpy' not in sys.modules "
                "and not any(m == 'src' or m.startswith('src.') for m in sys.modules)",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout!r} stderr={result.stderr!r}",
        )


if __name__ == "__main__":
    unittest.main()
