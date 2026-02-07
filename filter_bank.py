import numpy as np
from scipy.signal.windows import kaiser


def filter_bank(frame_T: np.ndarray, frame_type: str, win_type: str) -> np.ndarray:
    """Apply AAC filter bank (windowing + MDCT) to a 2048x2 frame.

    Returns an array of shape (1024, 2).
    For ESH, the 8 short-block MDCTs are stacked vertically in subframe order.
    """
    if frame_T.ndim != 2 or frame_T.shape != (2048, 2):
        raise ValueError("frame_T must be a 2048x2 matrix")
    if frame_type not in {"OLS", "ESH", "LSS", "LPS"}:
        raise ValueError("frame_type must be one of 'OLS', 'ESH', 'LSS', 'LPS'")
    if win_type not in {"KBD", "SIN"}:
        raise ValueError("win_type must be 'KBD' or 'SIN'")

    # Create the appropriate windows based on win_type.
    long_win = _make_window(2048, win_type, beta=6 * np.pi)
    short_win = _make_window(256, win_type, beta=4 * np.pi)

    # Apply the appropriate window and MDCT based on frame_type.
    if frame_type == "OLS":
        return _mdct_frame(frame_T, long_win)
    if frame_type == "LSS":
        lss_win = np.concatenate(
            [
                long_win[:1024],
                np.ones(448),
                short_win[128:],
                np.zeros(448),
            ]
        )
        return _mdct_frame(frame_T, lss_win)
    if frame_type == "LPS":
        lps_win = np.concatenate(
            [
                np.zeros(448),
                short_win[:128],
                np.ones(448),
                long_win[1024:],
            ]
        )
        return _mdct_frame(frame_T, lps_win)

    # ESH
    else:
        return _mdct_esh(frame_T, short_win)


def _make_window(N: int, win_type: str, beta: float) -> np.ndarray:
    # Sinusoid
    if win_type == "SIN":
        idx = np.arange(N)
        return np.sin(np.pi / N * (idx + 0.5))

    # Kaiser-Bessel Derived (KBD)
    else:
        half = N // 2
        w = kaiser(half + 1, beta)
        cumsum = np.cumsum(w)
        denom = cumsum[-1]
        left = np.sqrt(cumsum[:-1] / denom)
        right = left[::-1]
        return np.concatenate([left, right])


def _mdct_frame(frame_T: np.ndarray, window: np.ndarray) -> np.ndarray:
    # Apply window to each channel, then compute MDCT per channel.
    windowed = frame_T * window[:, None]
    out = np.zeros((1024, 2))
    for ch in range(2):
        # 2048-point MDCT -> 1024 coefficients.
        out[:, ch] = _mdct(windowed[:, ch])
    return out


def _mdct_esh(frame_T: np.ndarray, short_win: np.ndarray) -> np.ndarray:
    # For ESH, keep the middle 1152 samples and process 8 overlapping 256-sample blocks.
    out = np.zeros((1024, 2))
    mid = frame_T[448:1600]
    for ch in range(2):
        for i in range(8):
            start = i * 128
            block = mid[start:start + 256, ch]
            # Apply 256-point window, then MDCT -> 128 coefficients.
            windowed = block * short_win
            out[i * 128:(i + 1) * 128, ch] = _mdct(windowed)
    return out


def _mdct(x: np.ndarray) -> np.ndarray:
    # MDCT definition
    # X[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(2*pi/N * (n + n0) * (k + 1/2)), k=0..N/2-1
    N = x.shape[0]
    if N % 2 != 0:
        raise ValueError("MDCT input length must be even")
    k = np.arange(N // 2)[:, None]
    n = np.arange(N)[None, :]
    # Use standard MDCT time shift n0 = (N/2 + 1) / 2.
    n0 = (N / 2 + 1) / 2
    cos_arg = (2 * np.pi / N) * (n + n0) * (k + 0.5)
    return 2 * np.dot(np.cos(cos_arg), x)
