import numpy as np
from sliding_window import sliding_window
def opp_sliding_window_w_d(
    data_x: np.ndarray,
    data_y: np.ndarray,
    data_d: np.ndarray,
    window_size: int,
    step_size: int,
):
    """
    OPPORTUNITY sliding window with domain labels


    Args:
        data_x      : np.ndarray [T, F]
        data_y      : np.ndarray [T]
        data_d      : np.ndarray [T]
        window_size : int
        step_size   : int

    Returns:
        x_win : np.ndarray [N, window_size * F] (float32)
        y_win : np.ndarray [N] (uint8)
        d_win : np.ndarray [N] (uint8)
    """

    # features: slide over time, keep all features
    x_win = sliding_window(
        data_x,
        window_size=window_size,
        step_size=step_size,
        flatten=True,
    )

    # labels: take last timestep in each window
    y_windows = sliding_window(
        data_y,
        window_size=window_size,
        step_size=step_size,
        flatten=False
    )
    y_win = np.array([w[-1] for w in y_windows], dtype=np.uint8)

    # domain: take last timestep in each window
    d_windows = sliding_window(
        data_d,
        window_size=window_size,
        step_size=step_size,
        flatten=False,
    )
    d_win = np.array([w[-1] for w in d_windows], dtype=np.uint8)

    return (
        x_win.astype(np.float32),
        y_win,
        d_win,
    )
