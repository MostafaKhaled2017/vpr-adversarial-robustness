import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch import nn

from src import rank_eval
from src.evaluation_data import build_vpr_evaluation_dataset, load_vpr_test_dataset_module
from src.models import ModelBundle
from src.rank_attacks import RankAttackConfig, RankPGDAttack
from src.targets import RetrievalAttackBatch


class TinyDescriptorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]))

    def forward(self, inputs, queryflag=0):
        del queryflag
        pooled = inputs.mean(dim=(2, 3))
        return self.projection(pooled)


class RankEvalModelTypeTests(unittest.TestCase):
    def test_adapter_loader_builds_and_loads_each_checkpoint(self):
        loaded_paths = []

        def build(_args):
            return ModelBundle(model=TinyDescriptorModel(), descriptor_dim=2)

        def load_evaluation_weights(_model, checkpoint_path, _args):
            loaded_paths.append(checkpoint_path)

        adapter = SimpleNamespace(
            build=build,
            build_evaluation=None,
            load_weights=lambda *_args: None,
            load_evaluation_weights=load_evaluation_weights,
        )
        args = Namespace(
            model_type="fake",
            model_paths=["base.pth", "robust.pth"],
            model_tags=["base", "robust"],
            device="cpu",
        )

        with mock.patch.object(rank_eval, "get_model_adapter", return_value=adapter):
            models = rank_eval.load_models(args)

        self.assertEqual(loaded_paths, args.model_paths)
        self.assertEqual(list(models), args.model_tags)
        self.assertEqual(models["base"][1].features_dim, 2)
        self.assertEqual(rank_eval.attack_reference_tag(args), "base")

    def test_vpr_dataset_matches_reference_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_root = Path(temp_dir) / "sample" / "images" / "test"
            database_dir = test_root / "database"
            queries_dir = test_root / "queries"
            database_dir.mkdir(parents=True)
            queries_dir.mkdir(parents=True)
            image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
            database_path = database_dir / "@1.0@2.0@database.png"
            query_path = queries_dir / "@1.0@2.0@query.jpg"
            Image.fromarray(image).save(database_path)
            Image.fromarray(image).save(query_path)

            args = Namespace(
                eval_datasets_folder=temp_dir,
                val_positive_dist_threshold=25,
                resize=[320, 320],
                test_method="hard_resize",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                compatible = build_vpr_evaluation_dataset(args, "sample")
                upstream = load_vpr_test_dataset_module().TestDataset(
                    str(database_dir),
                    str(queries_dir),
                    positive_dist_threshold=25,
                    image_size=[320, 320],
                    use_labels=True,
                )

            compatible_image, compatible_index = compatible[0]
            upstream_image, upstream_index = upstream[0]

            self.assertEqual(compatible.database_num, upstream.num_database)
            self.assertEqual(compatible.queries_num, upstream.num_queries)
            self.assertEqual(compatible_index, upstream_index)
            self.assertEqual(tuple(compatible_image.shape), (3, 320, 320))
            self.assertTrue(torch.equal(compatible_image, upstream_image))
            np.testing.assert_array_equal(compatible.get_positives()[0], upstream.get_positives()[0])

    def test_seeded_rank_attack_is_repeatable(self):
        inputs = torch.full((1, 3, 4, 4), 0.25)
        targets = RetrievalAttackBatch(
            query_indices=torch.tensor([0]),
            clean_query_descriptors=torch.zeros(1, 2),
            positive_descriptors=torch.tensor([[0.5, 0.5]]),
            negative_descriptors=torch.tensor([[[-0.5, -0.5], [0.0, -0.5]]]),
        )
        config = RankAttackConfig(
            epsilon=0.1,
            steps=2,
            restarts=2,
            norm="linf",
            device="cpu",
        )

        torch.manual_seed(17)
        first = RankPGDAttack(TinyDescriptorModel(), config)(inputs, targets).adversarial
        torch.manual_seed(17)
        second = RankPGDAttack(TinyDescriptorModel(), config)(inputs, targets).adversarial

        self.assertTrue(torch.equal(first, second))

    def test_restart_initialization_changes_reproducibly_with_seed(self):
        inputs = torch.full((1, 3, 4, 4), 0.25)
        config = RankAttackConfig(epsilon=0.1, steps=1, restarts=2, norm="linf", device="cpu")
        attack = RankPGDAttack(TinyDescriptorModel(), config)

        torch.manual_seed(3)
        seed_three_first = attack._initial_inputs(inputs, restart_index=1)
        torch.manual_seed(3)
        seed_three_second = attack._initial_inputs(inputs, restart_index=1)
        torch.manual_seed(4)
        seed_four = attack._initial_inputs(inputs, restart_index=1)

        self.assertTrue(torch.equal(seed_three_first, seed_three_second))
        self.assertFalse(torch.equal(seed_three_first, seed_four))


if __name__ == "__main__":
    unittest.main()
