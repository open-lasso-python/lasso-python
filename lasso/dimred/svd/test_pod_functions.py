from unittest import TestCase
from lasso.dimred.svd.pod_functions import calculate_v_and_betas
import numpy as np
from typing import Tuple


class PodFunctionsTest(TestCase):
    def test_calculate_v_and_betas(self):
        """Verify svd works
        Test for:
        - returns V and B of correct shape
        - failes if dataset to small (1 sample)"""

        # random input for 1 sample, 5 timesteps, 100 nodes, 3 dimensions
        rand_samples = np.random.rand(1, 5, 100, 3)

        # should return error message string
        err_msg = calculate_v_and_betas(rand_samples)
        self.assertTrue(isinstance(err_msg, str))

        # random input for 5 samples, 5 timesteps, 100 nodes, 3 dimensions
        test_shape = (5, 5, 100, 3)
        samples, timesteps, nodes, dimensions = test_shape
        rand_samples = np.random.rand(samples, timesteps, nodes, dimensions)
        result = calculate_v_and_betas(rand_samples)

        # returns Tuple containing v_rob and betas
        self.assertTrue(isinstance(result, Tuple))

        v_rob, betas = result

        # v_rob and betas should both be numpy arrays
        self.assertTrue(isinstance(v_rob, np.ndarray))
        self.assertTrue(isinstance(betas, np.ndarray))

        # v_rob should be of shape (k_eigen, timesteps, nodes*dimensions)
        # k_eigen should be min(10, samples-1), so in this case k_eigen = samples-1 = 4
        k_eigen = min(10, samples - 1)
        self.assertEqual(v_rob.shape, (k_eigen, timesteps, nodes * dimensions))

        # betas should be of shape (samples, timesteps, k_eigen)
        self.assertEqual(betas.shape, (samples, timesteps, k_eigen))

        # v_rob and betas should result in difference in displacements of original result
        reshaped_samples = rand_samples.reshape(samples, timesteps, nodes * dimensions)

        delta_displ = reshaped_samples[:, :] - np.stack(
            [reshaped_samples[0, :] for _ in range(timesteps)]
        )

        recacl_displ = np.einsum("ktn, stk -> stn", v_rob, betas)

        # check if both original and recalc have the same shape
        self.assertEqual(delta_displ.shape, recacl_displ.shape)
