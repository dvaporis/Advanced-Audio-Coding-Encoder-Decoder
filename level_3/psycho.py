import numpy as np
from scipy.io import loadmat


def psycho(frame_T: np.ndarray, frame_type: str, frame_T_prev_1: np.ndarray | None = None, frame_T_prev_2: np.ndarray | None = None) -> np.ndarray:
	"""
	Psychoacoustic model for AAC encoding.
	
	Calculates the Signal to Mask Ratio (SMR) for each scale factor band.
	
	Parameters
	----------
	frame_T : np.ndarray
		Current frame in time domain (2048 samples)
	frame_type : str
		Frame type: "OLS", "ESH", "LSS", or "LPS"
	frame_T_prev_1 : np.ndarray | None, optional
		Previous frame in time domain (2048 samples). None for first frame.
	frame_T_prev_2 : np.ndarray | None, optional
		Frame before previous in time domain (2048 samples). None for first two frames.
	
	Returns
	-------
	SMR : np.ndarray
		Signal to Mask Ratio
		Shape: (42, 8) for ESH frames
		Shape: (69, 1) for other frame types
	"""
	if frame_type not in {"OLS", "ESH", "LSS", "LPS"}:
		raise ValueError("frame_type must be one of 'OLS', 'ESH', 'LSS', 'LPS'")
	
	# Load scale factor bands and precompute spreading function
	table = loadmat("TableB219.mat")
	
	if frame_type == "ESH":
		bands = table["B219b"]  # Short blocks (42 bands)
		n_bands = 42
		n_subframes = 8
		fft_size = 256
		spreading = _precompute_spreading_function(bands, n_bands)
	else:
		bands = table["B219a"]  # Long blocks (69 bands)
		n_bands = 69
		n_subframes = 1
		fft_size = 2048
		spreading = _precompute_spreading_function(bands, n_bands)
	
	# Process frames
	if frame_type == "ESH":
		# For short blocks, process each of 8 subframes
		SMR = np.zeros((n_bands, n_subframes))
		
		for sub in range(n_subframes):
			# Extract current subframe (256 samples)
			subframe_curr = frame_T[sub * 256:(sub + 1) * 256]
			
			# Get previous subframes for prediction
			# For sub 0: use subframes 6,7 from prev_frame_1 and current subframes 0,1
			# For sub i: use subframes i-1, i from current and i+1 from current
			if sub == 0 and frame_T_prev_1 is not None:
				subframe_prev1 = frame_T_prev_1[6 * 256:7 * 256]
				subframe_prev2 = frame_T_prev_1[7 * 256:8 * 256]
			elif sub == 1 and frame_T_prev_1 is not None:
				subframe_prev1 = frame_T[0 * 256:1 * 256]
				subframe_prev2 = frame_T_prev_1[7 * 256:8 * 256]
			elif sub > 0:
				subframe_prev1 = frame_T[(sub - 1) * 256:sub * 256]
				if sub > 1:
					subframe_prev2 = frame_T[(sub - 2) * 256:(sub - 1) * 256]
				elif frame_T_prev_1 is not None:
					subframe_prev2 = frame_T_prev_1[7 * 256:8 * 256]
				else:
					subframe_prev2 = subframe_prev1.copy()
			else:
				subframe_prev1 = subframe_curr.copy()
				subframe_prev2 = subframe_curr.copy()
			
			smr_sub = _compute_psychoacoustic_model(
				subframe_curr, subframe_prev1, subframe_prev2,
				bands, n_bands, fft_size, spreading
			)
			SMR[:, sub] = smr_sub
	else:
		# For long blocks, process full frame
		if frame_T_prev_1 is None:
			frame_prev1 = frame_T.copy()
		else:
			frame_prev1 = frame_T_prev_1
		
		if frame_T_prev_2 is None:
			frame_prev2 = frame_T.copy()
		else:
			frame_prev2 = frame_T_prev_2
		
		smr = _compute_psychoacoustic_model(
			frame_T, frame_prev1, frame_prev2,
			bands, n_bands, fft_size, spreading
		)
		SMR = smr.reshape(-1, 1)
	
	return SMR


def _precompute_spreading_function(bands: np.ndarray, n_bands: int) -> np.ndarray:
	"""
	Precompute spreading function for all band pairs.
	
	Parameters
	----------
	bands : np.ndarray
		Band table with columns: [index, wlow, whigh, bval, ...]
	n_bands : int
		Number of bands
	
	Returns
	-------
	spreading : np.ndarray
		Spreading function matrix of shape (n_bands, n_bands)
		spreading[i, j] = spreading_function(i, j)
	"""
	spreading = np.zeros((n_bands, n_bands))
	
	# Extract center frequencies (4th column, index 3)
	bval = bands[:, 3]
	
	for i in range(n_bands):
		for j in range(n_bands):
			spreading[i, j] = _spreading_function(i, j, bval)
	
	return spreading


def _spreading_function(i: int, j: int, bval: np.ndarray) -> float:
	"""
	Compute spreading function value for band i to band j.
	
	Parameters
	----------
	i : int
		Source band index
	j : int
		Target band index
	bval : np.ndarray
		Center frequencies of bands
	
	Returns
	-------
	float
		Spreading function value
	"""
	if i >= j:
		tmpx = 3.0 * (bval[j] - bval[i])
	else:
		tmpx = 1.5 * (bval[j] - bval[i])
	
	tmpz = 8.0 * np.minimum((tmpx - 0.5) ** 2 - 2.0 * (tmpx - 0.5), 0)
	tmpy = 15.811389 + 7.5 * (tmpx + 0.474) - 17.5 * np.sqrt(1.0 + (tmpx + 0.474) ** 2)
	
	if tmpy < -100:
		return 0.0
	else:
		return 10.0 ** ((tmpz + tmpy) / 10.0)


def _compute_psychoacoustic_model(
	frame_T_curr: np.ndarray,
	frame_T_prev1: np.ndarray,
	frame_T_prev2: np.ndarray,
	bands: np.ndarray,
	n_bands: int,
	fft_size: int,
	spreading: np.ndarray
) -> np.ndarray:
	"""
	Compute psychoacoustic model for a single frame/subframe.
	
	Implements steps 2-13 of the psychoacoustic model algorithm.
	"""
	# Step 2: Apply Hann window and compute FFT
	window = 0.5 - 0.5 * np.cos(np.pi * (np.arange(len(frame_T_curr)) + 0.5) / len(frame_T_curr))
	windowed_curr = frame_T_curr * window
	windowed_prev1 = frame_T_prev1 * window
	windowed_prev2 = frame_T_prev2 * window
	
	# Compute FFT (keep up to fft_size//2 bins)
	fft_curr = np.fft.rfft(windowed_curr, n=fft_size)
	fft_prev1 = np.fft.rfft(windowed_prev1, n=fft_size)
	fft_prev2 = np.fft.rfft(windowed_prev2, n=fft_size)
	
	r_curr = np.abs(fft_curr)
	f_curr = np.angle(fft_curr)
	r_prev1 = np.abs(fft_prev1)
	f_prev1 = np.angle(fft_prev1)
	r_prev2 = np.abs(fft_prev2)
	f_prev2 = np.angle(fft_prev2)
	
	# Step 3: Compute predictions
	r_pred = 2.0 * r_prev1 - r_prev2
	f_pred = 2.0 * f_prev1 - f_prev2
	
	# Step 4: Compute predictability measure c(w)
	real_diff = r_curr * np.cos(f_curr) - r_pred * np.cos(f_pred)
	imag_diff = r_curr * np.sin(f_curr) - r_pred * np.sin(f_pred)
	c_w = np.sqrt(real_diff ** 2 + imag_diff ** 2) / (r_curr + np.abs(r_pred) + 1e-10)
	
	# Step 5: Compute energy and weighted predictability for bands
	e_band = np.zeros(n_bands)
	c_band = np.zeros(n_bands)
	
	for b in range(n_bands):
		wlow = int(bands[b, 1])
		whigh = int(bands[b, 2])
		whigh = min(whigh, len(r_curr) - 1)
		
		if wlow >= len(r_curr):
			continue
		
		power = r_curr[wlow:whigh + 1] ** 2
		e_band[b] = np.sum(power)
		c_band[b] = np.sum(c_w[wlow:whigh + 1] * power)
	
	# Step 6: Combine with spreading function
	ecb = spreading.T @ e_band  # Shape: (n_bands,)
	ct = spreading.T @ c_band   # Shape: (n_bands,)
	
	# Normalize
	normalizer = np.sum(spreading, axis=0) + 1e-10
	cb = ct / (ecb + 1e-10)
	en = ecb / normalizer
	
	# Step 7: Compute tonality index
	tb = -0.299 - 0.43 * np.log(np.clip(cb, 1e-10, 1))
	tb = np.clip(tb, 0, 1)
	
	# Step 8: Compute required SNR
	TMN = 18.0  # Tone Masking Noise (dB)
	NMT = 6.0   # Noise Masking Tone (dB)
	SNR_required = tb * TMN + (1.0 - tb) * NMT
	
	# Step 9: Convert from dB to energy ratio
	bc = 10.0 ** (-SNR_required / 10.0)
	
	# Step 10: Compute energy threshold
	nb = en * bc
	
	# Step 11: Compute noise level with quiet threshold
	table = loadmat("TableB219.mat")
	
	if fft_size == 256:
		qsthr = table["B219b"][:, 4]  # Short table quiet threshold
	else:
		qsthr = table["B219a"][:, 4]  # Long table quiet threshold
	
	# Convert quiet threshold to energy
	eps = np.finfo(float).eps
	qsthr_energy = eps * fft_size ** 2 * 10.0 ** (qsthr / 10.0)
	
	npart = np.maximum(nb, qsthr_energy[:n_bands])
	
	# Step 12: Compute Signal to Mask Ratio (SMR)
	SMR = e_band / (npart + 1e-10)
	
	return SMR
