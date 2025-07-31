
import numpy as np

def validate_tensor_divergence(einstein_tensor, threshold=1e-5):
    """
    Validate tensor divergence to check approximate conservation (∇μ T^μν ≈ 0).

    Parameters:
        einstein_tensor (np.ndarray): 4D array representing Einstein tensor G^μν at each grid point.
        threshold (float): Maximum allowed divergence magnitude to pass validation.

    Returns:
        dict: Dictionary with statistics and pass/fail result.
    """
    divergence = np.gradient(einstein_tensor, axis=(0, 1, 2))
    divergence_magnitude = np.sqrt(sum(np.square(d) for d in divergence))

    mean_div = np.mean(divergence_magnitude)
    max_div = np.max(divergence_magnitude)

    result = {
        "mean_divergence": mean_div,
        "max_divergence": max_div,
        "threshold": threshold,
        "passes": max_div < threshold
    }

    return result
