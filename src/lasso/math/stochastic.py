import numpy as np
from scipy import stats


def jensen_shannon_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen Shannon Entropy

    Parameters
    ----------
    p: np.ndarray
        first probability distribution
    q: np.ndarray
        second probability distribution

    Returns
    -------
    js_divergence: float
        Jensen-Shannon divergence
    """
    p = np.asarray(p)
    q = np.asarray(q)
    # normalize
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2
    return (stats.entropy(p, m) + stats.entropy(q, m)) / 2
