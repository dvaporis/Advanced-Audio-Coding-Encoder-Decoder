import numpy as np
from scipy.io import loadmat


def aac_quantizer(frame_F: np.ndarray, frame_type: str, SMR: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	AAC quantizer with scale factor bands and psychoacoustic model.
	
	Implements AAC quantization with:
	- Scale factor bands from TableB219.mat
	- Perceptual quantization based on SMR
	- Global gain and scale factors per band
	
	Parameters
	----------
	frame_F : np.ndarray
		MDCT coefficients (TNS output).
		For ESH frames: shape (128, 8)
		For other frames: shape (1024, 1)
	frame_type : str
		Frame type: "OLS", "ESH", "LSS", or "LPS"
	SMR : np.ndarray
		Signal to Mask Ratio from psychoacoustic model
		Shape: (42, 8) for ESH, (69, 1) for others
	
	Returns
	-------
	S : np.ndarray
		Quantized MDCT coefficients, shape (1024, 1) for all frame types
	sfc : np.ndarray
		Scale factors per band
		Shape: (NB, 8) for ESH, (NB, 1) for others
	G : np.ndarray
		Global gain(s)
		Shape: (8,) for ESH, scalar for others
	"""
	if frame_type not in {"OLS", "ESH", "LSS", "LPS"}:
		raise ValueError("frame_type must be one of 'OLS', 'ESH', 'LSS', 'LPS'")
	
	# Load scale factor bands
	table = loadmat("TableB219.mat")
	if frame_type == "ESH":
		bands = table["B219b"]  # Short blocks
		n_bands = bands.shape[0]
		n_subframes = 8
	else:
		bands = table["B219a"]  # Long blocks
		n_bands = bands.shape[0]
		n_subframes = 1
	
	if frame_type == "ESH":
		# Process each subframe separately
		S_all = []
		sfc = np.zeros((n_bands, n_subframes))
		G = np.zeros(n_subframes)
		
		for sub in range(n_subframes):
			frame_sub = frame_F[:, sub]
			SMR_sub = SMR[:, sub]
			
			S_sub, sfc_sub, G_sub = _quantize_subframe(frame_sub, bands, SMR_sub, n_bands)
			
			S_all.append(S_sub)
			sfc[:, sub] = sfc_sub
			G[sub] = G_sub
		
		# Stack subframes into 1024×1
		S = np.concatenate(S_all).reshape(-1, 1)
	else:
		# Single frame
		frame_flat = frame_F.flatten()
		SMR_flat = SMR.flatten()
		
		S, sfc, G = _quantize_subframe(frame_flat, bands, SMR_flat, n_bands)
		
		S = S.reshape(-1, 1)
		sfc = sfc.reshape(-1, 1)
		G = np.array([G])
	
	return S, sfc, G


def _quantize_subframe(frame_F: np.ndarray, bands: np.ndarray, SMR: np.ndarray, n_bands: int) -> tuple[np.ndarray, np.ndarray, float]:
	"""Quantize a single subframe with scale factor bands."""
	n = len(frame_F)
	S = np.zeros(n, dtype=np.int32)
	sfc = np.zeros(n_bands)
	
	# Estimate global gain from frame energy
	frame_energy = np.sum(frame_F ** 2)
	if frame_energy > 0:
		G = 75 + 10 * np.log10(frame_energy / n)  # Empirical formula
	else:
		G = 0
	G = np.clip(G, 0, 255)
	
	# Quantize each scale factor band
	for b in range(n_bands):
		if b >= len(bands):
			break
		
		start = int(bands[b, 1])
		end = int(bands[b, 2])
		
		if start >= n:
			break
		end = min(end, n - 1)
		
		if end < start:
			continue
		
		# Extract band coefficients
		band_coeffs = frame_F[start:end + 1]
		band_energy = np.sum(band_coeffs ** 2)
		
		if band_energy == 0:
			sfc[b] = 0
			continue
		
		# Calculate scale factor based on SMR
		# Higher SMR = less masking = need finer quantization = smaller scale factor
		if b < len(SMR):
			smr_val = SMR[b]
			# Convert SMR to scale factor (empirical)
			sf = 100 - smr_val * 0.5
			sf = np.clip(sf, 0, 255)
		else:
			sf = 100
		
		sfc[b] = sf
		
		# Quantize band coefficients
		# AAC formula: q[i] = sign(x[i]) * int((|x[i]| / 2^((G-sf)/4))^(3/4) + 0.4054)
		Q = 2.0 ** ((G - sf) / 4.0)
		x_scaled = band_coeffs / (Q + 1e-10)
		x_abs = np.abs(x_scaled)
		x_pow = np.power(x_abs + 1e-10, 0.75)
		x_quant = np.floor(x_pow + 0.4054)
		band_quant = (np.sign(band_coeffs) * x_quant).astype(np.int32)
		
		S[start:end + 1] = band_quant
	
	return S, sfc, G
