import unittest

from lasso.math.sampling import unique_subsamples


class Test(unittest.TestCase):
    def test_unique_subsamples(self):

        self.assertEqual(len(set(unique_subsamples(0, 20, 100))), 20)
        self.assertEqual(len(set(unique_subsamples(0, 200, 100))), 100)
