import os
import tempfile
import unittest

import torch

from src.negative_bank import BANK_FILE_KEYS, NegativeBank, encode_bank, merge_bank_negatives, save_bank
from src.targets import RetrievalAttackBatch


class NegativeBankTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.descriptors = torch.randn(50, 6).half()
        self.place_ids = torch.arange(50) // 2
        self.queries = torch.randn(3, 6)

    def bank(self, **kwargs):
        return NegativeBank(self.descriptors, self.place_ids, device="cpu", **kwargs)

    def test_nearest_returns_true_nearest_rows(self):
        reference = self.descriptors.float()
        idx = torch.cdist(self.queries, reference).topk(4, largest=False).indices
        torch.testing.assert_close(self.bank().nearest(self.queries, 4), reference[idx])

    def test_nearest_excludes_the_query_place(self):
        queries = self.descriptors[[0, 3]].float()
        exclude = self.place_ids[[0, 3]]
        out = self.bank().nearest(queries, 48, exclude_place_ids=exclude)
        self.assertEqual(out.shape, (2, 48, 6))
        for b, place in enumerate(exclude.tolist()):
            excluded_rows = self.descriptors[self.place_ids == place].float()
            hits = (out[b][:, None, :] == excluded_rows[None]).all(-1)
            self.assertFalse(bool(hits.any()))

    def test_chunked_matches_unchunked(self):
        exclude = torch.tensor([0, 5, 9])
        small = self.bank(chunk_rows=7).nearest(self.queries, 10, exclude_place_ids=exclude)
        large = self.bank(chunk_rows=1000).nearest(self.queries, 10, exclude_place_ids=exclude)
        self.assertTrue(torch.equal(small, large))

    def test_cpu_bank_serves_queries_on_their_device(self):
        out = self.bank().nearest(self.queries, 3)
        self.assertEqual(out.device, self.queries.device)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.shape, (3, 3, 6))

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "bank.pt")
            save_bank(path, self.descriptors.float(), self.place_ids, model="supervlad", checkpoint="x.pth", image_size=[322, 322])
            loaded = NegativeBank.load(path, "cpu")
            stored = torch.load(path, weights_only=True)
        self.assertEqual(len(loaded), 50)
        self.assertEqual(loaded.descriptor_dim, 6)
        torch.testing.assert_close(loaded.nearest(self.queries, 5), self.bank().nearest(self.queries, 5))
        for key in BANK_FILE_KEYS:
            self.assertIn(key, stored)
        self.assertEqual(stored["descriptors"].dtype, torch.float16)
        self.assertEqual(stored["image_size"], [322, 322])

    def test_load_rejects_width_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bank.pt")
            save_bank(path, self.descriptors, self.place_ids, model="supervlad", checkpoint="x.pth", image_size=[322, 322])
            with self.assertRaises(ValueError) as ctx:
                NegativeBank.load(path, "cpu", expected_dim=5)
        self.assertIn("5", str(ctx.exception))
        self.assertIn("6", str(ctx.exception))

    def test_merge_keeps_shapes_and_other_fields(self):
        batch = RetrievalAttackBatch(
            query_indices=torch.tensor([0, 1]),
            clean_query_descriptors=torch.randn(2, 4),
            positive_descriptors=torch.randn(2, 2, 4),
            negative_descriptors=torch.randn(2, 3, 4),
            positive_mask=torch.tensor([[True, False], [True, True]]),
        )
        extra = torch.randn(2, 5, 4)
        merged = merge_bank_negatives(batch, extra)
        self.assertEqual(merged.negative_descriptors.shape, (2, 8, 4))
        self.assertTrue(torch.equal(merged.negative_descriptors[:, :3], batch.negative_descriptors))
        self.assertTrue(torch.equal(merged.negative_descriptors[:, 3:], extra))
        self.assertIs(merged.query_indices, batch.query_indices)
        self.assertIs(merged.clean_query_descriptors, batch.clean_query_descriptors)
        self.assertIs(merged.positive_descriptors, batch.positive_descriptors)
        self.assertIs(merged.positive_mask, batch.positive_mask)


    def test_encode_bank_stacks_fp16_descriptors_and_place_ids(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.queryflags = []

            def forward(self, images, queryflag=0):
                self.queryflags.append(queryflag)
                assert not self.training
                return images.flatten(1)[:, :5]

        model = TinyModel().train()
        images = torch.randn(4, 1, 3, 4, 4)
        loader = [(images[:2], torch.tensor([[7], [3]])), (images[2:], torch.tensor([[9], [1]]))]
        descriptors, place_ids = encode_bank(model, loader, device="cpu", mixed_precision=False)
        self.assertEqual(descriptors.dtype, torch.float16)
        self.assertEqual(descriptors.shape, (4, 5))
        torch.testing.assert_close(descriptors, images[:, 0].flatten(1)[:, :5].half())
        self.assertEqual(place_ids.dtype, torch.int64)
        self.assertEqual(place_ids.tolist(), [7, 3, 9, 1])
        self.assertEqual(model.queryflags, [1, 1])


if __name__ == "__main__":
    unittest.main()
