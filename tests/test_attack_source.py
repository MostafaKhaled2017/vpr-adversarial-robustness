"""Transfer evaluation: attacks generated on one model, every model evaluated on them."""

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src import rank_eval  # noqa: E402

MODELS = {"araft": ("model_a", "args_a"), "soup": ("model_s", "args_s"), "plain": ("model_p", "args_p")}


def argv(root: Path):
    for split in ("database", "queries"):
        (root / "msls" / "images" / "test" / split).mkdir(parents=True, exist_ok=True)
    paths = []
    for tag in MODELS:
        (root / f"{tag}.pth").touch()
        paths.append(str(root / f"{tag}.pth"))
    return [
        "--eval_datasets_folder", str(root), "--datasets", "msls", "--epsilons", "0.0685",
        "--model_type", "supervlad", "--device", "cpu", "--model_paths", *paths, "--model_tags", *MODELS,
    ]


class AttackSourceTest(unittest.TestCase):
    def test_default_is_one_group_per_model_in_order(self):
        args = Namespace(model_tags=list(MODELS), attack_source=None)
        groups = list(rank_eval.attack_groups(args, MODELS))
        self.assertEqual(groups, [(tag, {tag: bundle}) for tag, bundle in MODELS.items()])
        self.assertEqual(rank_eval.attack_reference_tag(args), "araft")

    def test_source_attacks_every_model(self):
        args = Namespace(model_tags=list(MODELS), attack_source="soup")
        self.assertEqual(list(rank_eval.attack_groups(args, MODELS)), [("soup", MODELS)])
        self.assertEqual(rank_eval.attack_reference_tag(args), "soup")

    def test_flag_is_parsed_and_checked(self):
        parser = rank_eval.build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            base = argv(Path(tmp))
            self.assertIsNone(rank_eval.finalize_arguments(parser.parse_args(base)).attack_source)
            args = rank_eval.finalize_arguments(parser.parse_args([*base, "--attack_source", "soup"]))
            self.assertEqual(args.attack_source, "soup")
            with self.assertRaisesRegex(ValueError, "--attack_source"):
                rank_eval.finalize_arguments(parser.parse_args([*base, "--attack_source", "fare"]))


if __name__ == "__main__":
    unittest.main()
