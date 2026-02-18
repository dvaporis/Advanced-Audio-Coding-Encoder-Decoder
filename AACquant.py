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
	"""
	Quantize a single subframe with scale factor bands according to AAC standard.
	
	Implements:
	- Initial approximation: α̂(b) = 16/3 * log2(max_k(X(k))^(3/4) / MQ)
	- Iterative refinement based on quantization error vs. threshold
	- DPCM encoding for scale factors
	"""
	n = len(frame_F)
	MQ = 8191  # Maximum quantization levels as per AAC standard
	
	# Convert SMR to thresholds T(b)
	# SMR is in dB, convert to power threshold
	# Higher SMR means less masking, so lower allowed quantization error
	T = np.zeros(n_bands)
	for b in range(n_bands):
		if b < len(SMR):
			# T(b) represents allowed quantization error power
			# Lower SMR (more masking) -> higher allowed error
			T[b] = 10 ** (-SMR[b] / 10.0)
		else:
			T[b] = 1e-6  # Very low threshold for bands without SMR data
	
	# Step 1: Calculate initial approximation α̂(b) for all bands
	# α̂(b) = 16/3 * log2(max_k(X(k))^(3/4) / MQ)
	max_coeff = np.max(np.abs(frame_F))
	if max_coeff > 0:
		alpha_hat = (16.0 / 3.0) * np.log2((max_coeff ** 0.75) / MQ)
	else:
		alpha_hat = 0.0
	
	# Initialize all scale factor gains with the initial approximation
	alpha = np.full(n_bands, alpha_hat)
	
	# Step 2: Iterative refinement - increase α(b) until Pe(b) reaches T(b)
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
		
		if np.max(np.abs(band_coeffs)) == 0:
			alpha[b] = 0
			continue
		
		# Iteratively increase α(b) while quantization error is below threshold
		max_iterations = 100  # Safety limit
		for iteration in range(max_iterations):
			# Quantize with current α(b)
			S_band = _quantize_band(band_coeffs, alpha[b])
			
			# Dequantize to calculate error
			X_hat_band = _dequantize_band(S_band, alpha[b])
			
			# Calculate quantization error power
			Pe = np.sum((band_coeffs - X_hat_band) ** 2)
			
			# If error is below threshold, we can increase α(b) (coarser quantization)
			if Pe < T[b]:
				# Check constraint: max difference between consecutive scale factors <= 60
				if b > 0 and abs(alpha[b] + 1 - alpha[b - 1]) > 60:
					break
				if b < n_bands - 1 and abs(alpha[b] + 1 - alpha[b + 1]) > 60:
					break
				
				alpha[b] += 1
			else:
				# Error exceeds threshold, stop refinement
				break
	
	# Clip alpha values to valid range
	alpha = np.clip(alpha, 0, 255)
	
	# Step 3: Set Global Gain G = α(0)
	G = alpha[0]
	
	# Step 4: Apply DPCM encoding for scale factors
	# sfc(b) = α(b) - α(b-1) for b > 0
	# sfc(0) = α(0) = G
	sfc = np.zeros(n_bands)
	sfc[0] = alpha[0]
	for b in range(1, n_bands):
		sfc[b] = alpha[b] - alpha[b - 1]
	
	# Step 5: Final quantization with optimized scale factors
	S = np.zeros(n, dtype=np.int32)
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
		
		band_coeffs = frame_F[start:end + 1]
		S_band = _quantize_band(band_coeffs, alpha[b])
		S[start:end + 1] = S_band
	
	return S, sfc, G


def _quantize_band(X: np.ndarray, alpha: float) -> np.ndarray:
	"""
	Quantize coefficients in a band using AAC formula.
	S(k) = sgn(X(k)) * int((|X(k)| × 2^(-α/4))^(3/4) + 0.4054)
	"""
	Q = 2.0 ** (alpha / 4.0)
	x_scaled = X / (Q + 1e-10)
	x_abs = np.abs(x_scaled)
	x_pow = np.power(x_abs + 1e-10, 0.75)
	x_quant = np.floor(x_pow + 0.4054)
	S = (np.sign(X) * x_quant).astype(np.int32)
	return S


def _dequantize_band(S: np.ndarray, alpha: float) -> np.ndarray:
	"""
	Dequantize coefficients in a band using AAC formula.
	X̂(k) = sgn(S(k)) * |S(k)|^(4/3) × 2^(α/4)
	"""
	Q = 2.0 ** (alpha / 4.0)
	s_abs = np.abs(S).astype(np.float64)
	x_pow = np.power(s_abs, 4.0 / 3.0)
	X_hat = np.sign(S) * Q * x_pow
	return X_hat
