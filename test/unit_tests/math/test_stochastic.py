import unittest

from lasso.math.stochastic import jensen_shannon_entropy


class Test(unittest.TestCase):
    def test_jensen_shannon_entropy(self):
        p1 = [0.5, 0.5, 0.0]
        p2 = [0, 0.1, 0.9]

        self.assertEqual(jensen_shannon_entropy(p1, p1), 0)
        self.assertAlmostEqual(jensen_shannon_entropy(p1, p2), 0.55797881790005399)
        self.assertAlmostEqual(jensen_shannon_entropy(p2, p1), 0.55797881790005399)
