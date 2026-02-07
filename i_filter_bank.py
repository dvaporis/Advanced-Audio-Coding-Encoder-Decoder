import numpy as np
from scipy.signal.windows import kaiser


def filter_bank(frame_F: np.ndarray, frame_type: str, win_type: str) -> np.ndarray:
	"""Inverse AAC filter bank (IMDCT + windowing) to recover 2048x2 frame."""
	if frame_type not in {"OLS", "ESH", "LSS", "LPS"}:
		raise ValueError("frame_type must be one of 'OLS', 'ESH', 'LSS', 'LPS'")
	if win_type not in {"KBD", "SIN"}:
		raise ValueError("win_type must be 'KBD' or 'SIN'")

	long_win = _make_window(2048, win_type, beta=6 * np.pi)
	short_win = _make_window(256, win_type, beta=4 * np.pi)

	if frame_type == "OLS":
		return _imdct_frame(frame_F, long_win)
	if frame_type == "LSS":
		lss_win = np.concatenate(
			[
				long_win[:1024],
				np.ones(448),
				short_win[128:],
				np.zeros(448),
			]
		)
		return _imdct_frame(frame_F, lss_win)
	if frame_type == "LPS":
		lps_win = np.concatenate(
			[
				np.zeros(448),
				short_win[:128],
				np.ones(448),
				long_win[1024:],
			]
		)
		return _imdct_frame(frame_F, lps_win)

	# ESH
	return _imdct_esh(frame_F, short_win)


def _make_window(N: int, win_type: str, beta: float) -> np.ndarray:
	# Sinusoid
	if win_type == "SIN":
		idx = np.arange(N)
		return np.sin(np.pi / N * (idx + 0.5))

	# Kaiser-Bessel Derived (KBD)
	half = N // 2
	w = kaiser(half + 1, beta)
	cumsum = np.cumsum(w)
	denom = cumsum[-1]
	left = np.sqrt(cumsum[:-1] / denom)
	right = left[::-1]
	return np.concatenate([left, right])


def _imdct_frame(frame_F: np.ndarray, window: np.ndarray) -> np.ndarray:
	if frame_F.ndim != 2 or frame_F.shape != (1024, 2):
		raise ValueError("frame_F must be a 1024x2 matrix")
	out = np.zeros((2048, 2))
	for ch in range(2):
		# 1024-point IMDCT -> 2048 samples.
		out[:, ch] = _imdct(frame_F[:, ch])
	# Apply synthesis window.
	return out * window[:, None]


def _imdct_esh(frame_F: np.ndarray, short_win: np.ndarray) -> np.ndarray:
	if frame_F.ndim != 2 or frame_F.shape != (1024, 2):
		raise ValueError("frame_F must be a 1024x2 matrix for ESH")
	out = np.zeros((2048, 2))
	mid = np.zeros((1152, 2))

	for ch in range(2):
		for i in range(8):
			coeffs = frame_F[i * 128:(i + 1) * 128, ch]
			block = _imdct(coeffs) * short_win
			start = i * 128
			mid[start:start + 256, ch] += block

	# Place the reconstructed middle 1152 samples into the 2048 frame.
	out[448:1600, :] = mid
	return out


def _imdct(X: np.ndarray) -> np.ndarray:
	# Inverse MDCT corresponding to the forward definition in filter_bank.
	M = X.shape[0]
	N = 2 * M
	k = np.arange(M)[None, :]
	n = np.arange(N)[:, None]
	n0 = (N / 2 + 1) / 2
	cos_arg = (2 * np.pi / N) * (n + n0) * (k + 0.5)
	return (2 / N) * np.dot(np.cos(cos_arg), X)