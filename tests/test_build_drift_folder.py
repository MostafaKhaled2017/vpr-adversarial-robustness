"""Sampling unlabeled city images into the drift constraint's folder."""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.drift_images import list_images  # noqa: E402

spec = importlib.util.spec_from_file_location("build_drift_folder", REPO_ROOT / "scripts" / "build_drift_folder.py")
build_drift_folder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_drift_folder)


class BuildDriftFolderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cities = []
        for city, count in (("amman", 4), ("manila", 3)):
            for split in ("database", "query"):
                folder = self.root / city / split / "images"
                folder.mkdir(parents=True)
                for index in range(count):
                    Image.fromarray(np.full((4, 4, 3), index, dtype=np.uint8)).save(folder / f"{index}.jpg")
            (self.root / city / "postprocessed.csv").write_text("not an image")
            self.cities.append(self.root / city)

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            build_drift_folder.main([*map(str, args)])
        return out.getvalue()

    def test_draw_is_seeded_and_without_replacement(self):
        first = build_drift_folder.sample_images(self.cities, 5, seed=0)
        self.assertEqual(first, build_drift_folder.sample_images(self.cities, 5, seed=0))
        self.assertEqual(len({name for _, name in first}), 5)
        self.assertNotEqual(first, build_drift_folder.sample_images(self.cities, 5, seed=1))
        self.assertEqual(len(build_drift_folder.sample_images(self.cities, 100, seed=0)), 14)

    def test_folder_holds_the_images_and_their_sources(self):
        out = self.root / "ood_drift"
        self.assertIn("5 images", self.run_main(*self.cities, "--out", out, "--count", 5))
        images = list_images(out)
        self.assertEqual(len(images), 5)
        self.assertTrue(all(path.name.split("_", 1)[0] in {"amman", "manila"} for path in images))
        names = {name for _, name in build_drift_folder.sample_images(self.cities, 100, 0)}
        self.assertIn("amman_database_images_0.jpg", names)
        self.assertNotIn("amman_postprocessed.csv", names)
        sources = (out / "sources.txt").read_text().splitlines()
        self.assertEqual(len(sources), 5)
        name, source = sources[0].split("\t")
        self.assertTrue((out / name).is_file() and Path(source).is_file())

    def test_symlinks(self):
        out = self.root / "linked"
        self.run_main(*self.cities, "--out", out, "--count", 2, "--symlink")
        self.assertTrue(all(path.is_symlink() for path in list_images(out)))

    def test_refusals(self):
        held_out = self.root / "cph"
        held_out.mkdir()
        empty = self.root / "empty_city"
        empty.mkdir()
        busy = self.root / "busy"
        busy.mkdir()
        (busy / "x.jpg").touch()
        for args in ((held_out, "--out", self.root / "o1"), (empty, "--out", self.root / "o2"),
                     (self.cities[0], "--out", busy), (self.cities[0], "--out", self.root / "o3", "--count", 0)):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.run_main(*args)
        self.assertFalse((self.root / "o1").exists())


if __name__ == "__main__":
    unittest.main()
