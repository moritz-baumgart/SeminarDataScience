import numpy as np

def sliding_window(
    x: np.ndarray,
    window_size: int,
    step_size: int,
    flatten: bool = True,
):
    """
    Sliding window over time dimension (axis=0), NumPy version

    Args:
        x           : np.ndarray [T, F] or [T]
        window_size : length of window
        step_size   : stride between windows
        flatten     : if True -> [N, window_size * F]
                      else    -> [N, window_size, F]

    Returns:
        np.ndarray of windows
    """

    # ensure 2D input
    if x.ndim == 1:
        x = x[:, None]  # [T, 1]

    T, F = x.shape

    if T < window_size:
        raise ValueError("window_size larger than sequence length")

    windows = []

    for start in range(0, T - window_size + 1, step_size):
        w = x[start : start + window_size]  # [window_size, F]
        windows.append(w)

    windows = np.stack(windows, axis=0)  # [N, window_size, F]

    if flatten:
        windows = windows.reshape(windows.shape[0], -1)

    return windows
