import numpy as np

def evaluate(data):
    vectors = np.array(data["vectors"], dtype=np.float64)
    assert vectors.shape == (50, 3), f"Expected (50, 3), got {vectors.shape}"
    assert np.all(np.isfinite(vectors)), "vectors must be finite"
    scale = np.max(np.abs(vectors), axis=1, keepdims=True)
    assert np.all(scale > 0), "vectors must be non-zero"
    # Pre-scaling avoids under/overflow in the norm for tiny or huge inputs.
    vectors = vectors / scale
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    diffs = vectors[:, None, :] - vectors[None, :, :]
    dist_sq = np.sum(diffs**2, axis=2)
    iu = np.triu_indices(50, k=1)
    dists = np.sqrt(dist_sq[iu])
    return float(np.min(dists))