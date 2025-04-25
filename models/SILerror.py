import numpy as np
def scale_invariant_log_error(y_true, y_pred) -> float:
    """
    Compute the scale-invariant log error used in on the KITTI dataset.
    y_true: ground truth values
    y_pred: predicted values

    Returns float: scale-invariant log error
    """

    # Convert to numpy arrays if they are not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute the scale-invariant log error
    log_error = np.log(np.maximum(y_pred, 1e-6)) - np.log(np.maximum(y_true, 1e-6))
    n = len(y_true)
    
    sil_error = (np.sum(log_error ** 2) / n) - (np.sum(log_error) / n)**2
    return sil_error