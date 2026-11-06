"""Weight average and backbone transplant of two checkpoints."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


soup = load_script("soup")
transplant = load_script("transplant")


def state(seed):
    generator = torch.Generator().manual_seed(seed)
    return OrderedDict(
        [
            ("backbone.weight", torch.randn(3, 4, generator=generator)),
            ("head.half", torch.randn(5, generator=generator).half()),
            ("head.count", torch.tensor([seed, seed + 1], dtype=torch.int64)),
        ]
    )


class SoupTest(unittest.TestCase):
    def setUp(self):
        self.a, self.b = state(1), state(2)

    def test_alpha_zero_and_one_reproduce_the_constituents(self):
        for alpha, expected in ((0.0, self.a), (1.0, self.b)):
            with self.subTest(alpha=alpha):
                mixed, interpolated = soup.soup_state_dicts(self.a, self.b, alpha)
                self.assertEqual(interpolated, 2)
                self.assertEqual(list(mixed), list(self.a))
                for key in ("backbone.weight", "head.half"):
                    self.assertTrue(torch.equal(mixed[key], expected[key]), key)
                    self.assertEqual(mixed[key].dtype, self.a[key].dtype)

    def test_half_is_the_mean_in_a_dtype(self):
        mixed, _ = soup.soup_state_dicts(self.a, self.b, 0.5)
        for key in ("backbone.weight", "head.half"):
            expected = ((self.a[key].float() + self.b[key].float()) / 2).to(self.a[key].dtype)
            self.assertEqual(mixed[key].dtype, self.a[key].dtype)
            self.assertTrue(torch.allclose(mixed[key].float(), expected.float(), atol=1e-6), key)

    def test_integer_buffer_is_copied_from_a(self):
        for alpha in (0.0, 0.5, 1.0):
            mixed, _ = soup.soup_state_dicts(self.a, self.b, alpha)
            self.assertTrue(torch.equal(mixed["head.count"], self.a["head.count"]))

    def test_unequal_keys_are_refused(self):
        b = OrderedDict(self.b)
        b["head.extra"] = torch.zeros(1)
        with self.assertRaises(SystemExit) as raised:
            soup.soup_state_dicts(self.a, b, 0.5)
        self.assertIn("head.extra", str(raised.exception))

    def test_unequal_shapes_are_refused(self):
        b = OrderedDict(self.b)
        b["backbone.weight"] = torch.zeros(4, 4)
        with self.assertRaises(SystemExit):
            soup.soup_state_dicts(self.a, b, 0.5)

    def test_refused_pair_writes_nothing_and_main_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            torch.save({"model_state_dict": OrderedDict(("module." + k, v) for k, v in self.a.items())}, tmp / "a.pth")
            torch.save({"model_state_dict": self.b}, tmp / "b.pth")
            bad = OrderedDict(self.b, extra=torch.zeros(1))
            torch.save({"model_state_dict": bad}, tmp / "bad.pth")
            with self.assertRaises(SystemExit):
                soup.main([str(tmp / "a.pth"), str(tmp / "bad.pth"), "0.5", str(tmp / "out_bad.pth")])
            self.assertFalse((tmp / "out_bad.pth").exists())

            with contextlib.redirect_stdout(io.StringIO()) as out:
                soup.main([str(tmp / "a.pth"), str(tmp / "b.pth"), "0", str(tmp / "out.pth")])
            self.assertIn("2 float tensors interpolated", out.getvalue())
            written = torch.load(tmp / "out.pth", weights_only=False)["model_state_dict"]
            for key, value in self.a.items():
                self.assertTrue(torch.equal(written[key], value), key)


class TransplantTest(unittest.TestCase):
    def setUp(self):
        self.backbone, self.head = state(1), state(2)

    def test_only_backbone_tensors_move(self):
        mixed, from_backbone, from_head = transplant.transplant_state_dicts(self.backbone, self.head)
        self.assertEqual((from_backbone, from_head), (1, 2))
        self.assertEqual(list(mixed), list(self.head))
        self.assertTrue(torch.equal(mixed["backbone.weight"], self.backbone["backbone.weight"]))
        self.assertTrue(torch.equal(mixed["head.half"], self.head["head.half"]))
        self.assertTrue(torch.equal(mixed["head.count"], self.head["head.count"]))

    def test_backbone_from_a_model_without_the_head(self):
        donor = OrderedDict([("backbone.weight", self.backbone["backbone.weight"])])
        mixed, _, _ = transplant.transplant_state_dicts(donor, self.head)
        self.assertTrue(torch.equal(mixed["backbone.weight"], donor["backbone.weight"]))

    def test_extra_backbone_key_is_refused(self):
        donor = OrderedDict(self.backbone)
        donor["backbone.extra"] = torch.zeros(1)
        with self.assertRaises(SystemExit) as raised:
            transplant.transplant_state_dicts(donor, self.head)
        self.assertIn("backbone.extra", str(raised.exception))

    def test_head_with_a_backbone_key_the_donor_lacks_is_refused(self):
        head = OrderedDict(self.head)
        head["backbone.extra"] = torch.zeros(1)
        with self.assertRaises(SystemExit) as raised:
            transplant.transplant_state_dicts(self.backbone, head)
        self.assertIn("backbone.extra", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
