import numpy as np
from scipy.io import loadmat


def i_aac_quantizer(S: np.ndarray, sfc: np.ndarray, G: np.ndarray | float, frame_type: str) -> np.ndarray:
	"""
	Inverse AAC quantizer - reconstructs MDCT coefficients from quantized values.
	
	Parameters
	----------
	S : np.ndarray
		Quantized MDCT coefficients, shape (1024, 1) for all frame types
	sfc : np.ndarray
		Scale factors per band
		Shape: (NB, 8) for ESH, (NB, 1) for others
	G : np.ndarray or float
		Global gain of the current frame
		- For EIGHT SHORT SEQUENCE (ESH): shape (8,) - one value per subframe (1 × 8)
		- For other frame types: scalar - one value for all coefficients
	frame_type : str
		Frame type: "OLS", "ESH", "LSS", or "LPS"
	
	Returns
	-------
	frame_F : np.ndarray
		Reconstructed MDCT coefficients
		For ESH frames: shape (128, 8)
		For other frames: shape (1024, 1)
	"""
	if frame_type not in {"OLS", "ESH", "LSS", "LPS"}:
		raise ValueError("frame_type must be one of 'OLS', 'ESH', 'LSS', 'LPS'")
	
	# Load scale factor bands
	table = loadmat("TableB219.mat")
	if frame_type == "ESH":
		bands = table["B219b"]  # Short blocks
		n_bands = bands.shape[0]
		n_subframes = 8
		frame_len = 128
	else:
		bands = table["B219a"]  # Long blocks
		n_bands = bands.shape[0]
		n_subframes = 1
		frame_len = 1024
	
	S_flat = S.flatten()
	
	# Handle G parameter - convert to proper format
	if isinstance(G, (int, float)):
		if frame_type == "ESH":
			# Broadcast scalar to all 8 subframes
			G_array = np.full(8, G)
		else:
			G_val = float(G)
	else:
		G = np.asarray(G).flatten()
		if frame_type == "ESH":
			if len(G) != 8:
				raise ValueError(f"For ESH frames, G must have 8 values, got {len(G)}")
			G_array = G.astype(np.float64)
		else:
			if len(G) != 1:
				raise ValueError(f"For non-ESH frames, G must be scalar or single-element array, got {len(G)}")
			G_val = float(G[0])
	
	if frame_type == "ESH":
		# Process each subframe separately
		frame_F = np.zeros((frame_len, n_subframes))
		
		for sub in range(n_subframes):
			start_idx = sub * frame_len
			end_idx = start_idx + frame_len
			S_sub = S_flat[start_idx:end_idx]
			
			sfc_sub = sfc[:, sub]
			
			G_sub = float(G_array[sub])
			
			frame_sub = _dequantize_subframe(S_sub, sfc_sub, G_sub, bands, n_bands, frame_len)
			frame_F[:, sub] = frame_sub
	else:
		# Single frame - G should be scalar
		frame_F = _dequantize_subframe(S_flat, sfc[:, 0], G_val, bands, n_bands, frame_len)
		frame_F = frame_F.reshape(-1, 1)
	
	return frame_F


def _dequantize_subframe(S: np.ndarray, sfc: np.ndarray, G: float, bands: np.ndarray, n_bands: int, frame_len: int) -> np.ndarray:
	"""Dequantize a single subframe with scale factor bands."""
	frame_F = np.zeros(frame_len)
	
	# Dequantize each scale factor band
	for b in range(n_bands):
		if b >= len(bands):
			break
		
		start = int(bands[b, 1])
		end = int(bands[b, 2])
		
		if start >= frame_len:
			break
		end = min(end, frame_len - 1)
		
		if end < start:
			continue
		
		# Get scale factor for this band
		sf = sfc[b] if b < len(sfc) else 0
		
		# Inverse quantization formula: x[i] = sign(q[i]) * 2^((G-sf)/4) * |q[i]|^(4/3)
		Q = 2.0 ** ((G - sf) / 4.0)
		
		q = S[start:end + 1]
		q_sign = np.sign(q)
		q_abs = np.abs(q).astype(np.float64)
		
		# Apply 4/3 power law
		x_pow = np.power(q_abs, 4.0 / 3.0)
		
		# Scale and apply sign
		frame_F[start:end + 1] = q_sign * Q * x_pow
	
	return frame_F
