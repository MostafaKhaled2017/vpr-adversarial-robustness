import unittest

import torch

from src.losses import compute_relational_loss


class RelationalLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_zero_when_adversarial_equals_reference(self):
        r, n = torch.randn(3, 4), torch.randn(3, 5, 4)
        self.assertAlmostEqual(float(compute_relational_loss(r, r.clone(), n, 0.1)), 0.0, places=6)

    def test_positive_and_differentiable_otherwise(self):
        r, n = torch.randn(3, 4), torch.randn(3, 5, 4)
        a = (r + 0.5).requires_grad_(True)
        loss = compute_relational_loss(r, a, n, 0.1)
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_matches_explicit_kl(self):
        r, a, n = torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 5, 4)
        temperature = 0.5
        p = torch.softmax(-((r.unsqueeze(1) - n) ** 2).sum(dim=2) / temperature, dim=1)
        q = torch.softmax(-((a.unsqueeze(1) - n) ** 2).sum(dim=2) / temperature, dim=1)
        expected = (p * (p.log() - q.log())).sum(dim=1).mean()
        actual = compute_relational_loss(r, a, n, temperature)
        self.assertAlmostEqual(float(actual), float(expected), places=5)

    def test_temperature_changes_the_value(self):
        r, a, n = torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 5, 4)
        self.assertNotAlmostEqual(
            float(compute_relational_loss(r, a, n, 0.1)),
            float(compute_relational_loss(r, a, n, 1.0)),
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
