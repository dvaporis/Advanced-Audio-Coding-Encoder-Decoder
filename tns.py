import numpy as np
from scipy.io import loadmat
from numpy.polynomial.polynomial import Polynomial


def tns(frame_F_in: np.ndarray, frame_type: str) -> tuple[np.ndarray, np.ndarray]:
	if frame_type not in {"OLS", "ESH", "LSS", "LPS"}:
		raise ValueError("frame_type must be one of 'OLS', 'ESH', 'LSS', 'LPS'")

	if frame_type == "ESH":
		if frame_F_in.shape != (128, 8):
			raise ValueError("ESH frame_F_in must be 128x8")
		bands = _load_bands(short=True)
		frame_F_out = np.zeros_like(frame_F_in)
		tns_coeffs = np.zeros((4, 8))
		for i in range(8):
			coeffs, tns_frame = _process_subframe(frame_F_in[:, i], bands)
			tns_coeffs[:, i] = coeffs
			frame_F_out[:, i] = tns_frame
		return frame_F_out, tns_coeffs

	if frame_F_in.shape != (1024, 1):
		raise ValueError("Non-ESH frame_F_in must be 1024x1")
	bands = _load_bands(short=False)
	coeffs, tns_frame = _process_subframe(frame_F_in[:, 0], bands)
	tns_coeffs = coeffs.reshape(4, 1)
	frame_F_out = tns_frame.reshape(1024, 1)
	return frame_F_out, tns_coeffs

# Helper function to load the band tables from TableB219.mat for both short and long blocks. The function checks the structure of the loaded data to ensure it contains the expected band information.

def _load_bands(short: bool) -> np.ndarray:
	table = loadmat("TableB219.mat")
	key = "B219b" if short else "B219a"
	bands = table[key]
	if bands.ndim != 2 or bands.shape[1] < 3:
		raise ValueError("TableB219.mat must contain band tables with w_low and w_high")
	return bands

# Process a single subframe (either a full frame for non-ESH or a short block for ESH) to compute TNS coefficients and apply the TNS filter. Using the below helper functions.

def _process_subframe(frame_F: np.ndarray, bands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	xw, _ = _normalize_bands(frame_F, bands)
	lpc = _lpc_coeffs(xw, order=4)
	q = _quantize_coeffs(lpc, step=0.1, bits=4)

	if not _is_stable(q):
		q = np.zeros_like(q)

	y = _apply_tns(frame_F, q)
	return q, y

# Helper function for the normalization step of TNS.

def _normalize_bands(frame_F: np.ndarray, bands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	n = frame_F.shape[0]
	Sw = np.zeros(n)
	# Compute the energy in each band and store it in Sw for normalization.
	for row in bands:
		start = int(row[1])
		end = int(row[2])
		if start >= n:
			continue
		end = min(end, n - 1)
		if end < start:
			continue
		band = frame_F[start:end + 1]
		energy = np.sqrt(np.sum(band ** 2))
		Sw[start:end + 1] = energy

	# Smooth the Sw values across bands to avoid sharp transitions.
	for k in range(n - 2, -1, -1):
		Sw[k] = (Sw[k] + Sw[k + 1]) / 2
	for k in range(1, n):
		Sw[k] = (Sw[k] + Sw[k - 1]) / 2
	
	# Avoid division by zero by replacing zeros in Sw with a small value.
	denom = np.where(Sw > 0, Sw, 1.0)
	Xw = frame_F / denom
	return Xw, Sw

# Compute LPC coefficients using autocorrelation method.

def _lpc_coeffs(x: np.ndarray, order: int) -> np.ndarray:
	if x.ndim != 1:
		x = x.reshape(-1)
	n = x.shape[0]
	if n <= order:
		return np.zeros(order)

    # Compute autocorrelation r[k] for k=0..order.
	r = np.zeros(order + 1)
	for k in range(order + 1):
		r[k] = np.dot(x[k:], x[:n - k])

    # Define the autocorrelation matrix R.
	R = np.zeros((order, order))
	for i in range(order):
		for j in range(order):
			lag = i - j
			R[i, j] = r[lag] if lag >= 0 else r[-lag]
	rvec = r[1:order + 1]

    # Solve the normal equations Ra = rvec for LPC coefficients a. If R is singular, return zeros.
	try:
		a = np.linalg.solve(R, rvec)
	except np.linalg.LinAlgError:
		a = np.zeros(order)
	return a

# Quantize LPC coefficients to a specified step size and bit depth.

def _quantize_coeffs(a: np.ndarray, step: float, bits: int) -> np.ndarray:
	max_val = (2 ** (bits - 1) - 1) * step
	q = np.round(a / step) * step
	return np.clip(q, -max_val, max_val)

# Helper function to check if the quantized LPC coefficients correspond to a stable filter.

def _is_stable(a: np.ndarray) -> bool:
	if np.allclose(a, 0):
		return True
	p = a.shape[0]
	# 1 - a1 z^-1 - ... - ap z^-p
	coeffs = np.concatenate(([1.0], -a))
	roots_inv = Polynomial(coeffs).roots()
	if np.any(np.isclose(roots_inv, 0.0)):
		return False
	roots = 1.0 / roots_inv
	return np.all(np.abs(roots) < 1.0)

# Helper function to apply the TNS filter to the MDCT coefficients.

def _apply_tns(frame_F: np.ndarray, a: np.ndarray) -> np.ndarray:
	if np.allclose(a, 0):
		return frame_F.copy()
	b = np.concatenate(([1.0], -a))
	y = np.convolve(frame_F, b, mode="full")[: frame_F.shape[0]]
	return y
