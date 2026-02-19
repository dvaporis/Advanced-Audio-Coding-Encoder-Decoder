import numpy as np


def i_tns(frame_F_in: np.ndarray, frame_type: str, tns_coeffs: np.ndarray) -> np.ndarray:
	"""Inverse TNS: reconstruct the original MDCT coefficients."""
	if frame_type not in {"OLS", "ESH", "LSS", "LPS"}:
		raise ValueError("frame_type must be one of 'OLS', 'ESH', 'LSS', 'LPS'")

    # Validate input shapes based on frame_type and tns_coeffs dimensions and apply \
    # the inverse TNS filter to reconstruct the original MDCT coefficients.
	if frame_type == "ESH":
		if frame_F_in.shape != (128, 8):
			raise ValueError("ESH frame_F_in must be 128x8")
		if tns_coeffs.shape != (4, 8):
			raise ValueError("ESH tns_coeffs must be 4x8")
		frame_F_out = np.zeros_like(frame_F_in)
		for i in range(8):
			frame_F_out[:, i] = _apply_itns(frame_F_in[:, i], tns_coeffs[:, i])
		return frame_F_out

	if frame_F_in.shape != (1024, 1):
		raise ValueError("Non-ESH frame_F_in must be 1024x1")
	if tns_coeffs.shape not in {(4, 1), (4,)}:
		raise ValueError("Non-ESH tns_coeffs must be 4x1 or length-4")
	coeffs = tns_coeffs.reshape(-1)
	frame_F_out = _apply_itns(frame_F_in[:, 0], coeffs).reshape(1024, 1)
	return frame_F_out


# Helper function to apply the inverse TNS filter to a single subframe. 
# This function implements the inverse of the TNS filtering process, which 
# is a recursive filter defined by the quantized coefficients. The function 
# checks if all coefficients are zero (indicating no filtering) and returns 
# the input unchanged in that case. Otherwise, it applies the inverse filter 
# using a loop to reconstruct the original MDCT coefficients from the filtered ones.

def _apply_itns(frame_F: np.ndarray, a: np.ndarray) -> np.ndarray:
	if np.allclose(a, 0):
		return frame_F.copy()
	order = a.shape[0]
	out = np.zeros_like(frame_F)
	# Invert y[n] = x[n] - sum a[k] x[n-k]  =>  x[n] = y[n] + sum a[k] x[n-k]
	for n in range(frame_F.shape[0]):
		acc = frame_F[n]
		for k in range(1, order + 1):
			idx = n - k
			if idx < 0:
				break
			acc += a[k - 1] * out[idx]
		out[n] = acc
	return out