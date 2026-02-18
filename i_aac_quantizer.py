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
		DPCM-encoded scale factors per band
		Shape: (NB, 8) for ESH, (NB, 1) for others
		Note: sfc contains differences (sfc[0] = α[0], sfc[b] = α[b] - α[b-1] for b > 0)
	G : np.ndarray or float
		Global gain of the current frame (should match sfc[0])
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
	
	if frame_type == "ESH":
		# Handle G parameter for ESH frames
		if isinstance(G, (int, float)):
			G_array = np.full(8, float(G))
		else:
			G = np.asarray(G).flatten()
			if len(G) != 8:
				raise ValueError(f"For ESH frames, G must have 8 values, got {len(G)}")
			G_array = G.astype(np.float64)
		
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
		# Handle G parameter for non-ESH frames
		if isinstance(G, (int, float)):
			G_val = float(G)
		else:
			G = np.asarray(G).flatten()
			if len(G) != 1:
				raise ValueError(f"For non-ESH frames, G must be scalar or single-element array, got {len(G)}")
			G_val = float(G[0])
		
		# Single frame
		frame_F = _dequantize_subframe(S_flat, sfc[:, 0], G_val, bands, n_bands, frame_len)
		frame_F = frame_F.reshape(-1, 1)
	
	return frame_F


def _dequantize_subframe(S: np.ndarray, sfc: np.ndarray, G: float, bands: np.ndarray, n_bands: int, frame_len: int) -> np.ndarray:
	"""
	Dequantize a single subframe with scale factor bands.
	
	Decodes DPCM-encoded scale factors and applies inverse quantization.
	"""
	frame_F = np.zeros(frame_len)
	
	# Step 1: Decode DPCM to reconstruct α(b) values
	# sfc(0) = α(0) = G
	# sfc(b) = α(b) - α(b-1) for b > 0
	alpha = np.zeros(n_bands)
	alpha[0] = sfc[0]  # Should equal G
	for b in range(1, n_bands):
		alpha[b] = alpha[b - 1] + sfc[b]
	
	# Step 2: Dequantize each scale factor band using reconstructed α(b)
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
		
		# Get quantized values for this band
		q = S[start:end + 1]
		
		# Inverse quantization formula: X̂(k) = sgn(S(k)) * |S(k)|^(4/3) × 2^(α/4)
		Q = 2.0 ** (alpha[b] / 4.0)
		
		q_sign = np.sign(q)
		q_abs = np.abs(q).astype(np.float64)
		
		# Apply 4/3 power law
		x_pow = np.power(q_abs, 4.0 / 3.0)
		
		# Scale and apply sign
		frame_F[start:end + 1] = q_sign * Q * x_pow
	
	return frame_F
