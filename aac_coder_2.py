import numpy as np
import soundfile as sf

from SSC import SSC
from filter_bank import filter_bank
from tns import tns


def aac_coder_2(filename_in: str) -> list:
	"""AAC encoder level 2: apply TNS and return list of frame dictionaries."""
	x, fs = sf.read(filename_in, always_2d=True)
	if fs != 48000:
		raise ValueError("Input must be 48 kHz")

	# Ensure float64 processing.
	if not np.issubdtype(x.dtype, np.floating):
		x = x.astype(np.float64)
	else:
		x = x.astype(np.float64)

	if x.ndim != 2 or x.shape[1] != 2:
		raise ValueError("Input must be stereo (2 channels)")

	# Pad with 1024 zeros at start and end for first/last frame processing.
	pad = np.zeros((1024, 2))
	x_padded = np.vstack([pad, x, pad])

	# Define frame length and hop size (overlap of 50%).
	frame_len = 2048
	hop = 1024
	num_frames = (x_padded.shape[0] - frame_len) // hop + 1

	# Initialize list to hold AAC frame dictionaries and previous frame type.
	aac_seq_2 = []
	prev_frame_type = "OLS"
	win_type = "KBD"

	# Process each frame and determine frame type using SSC.
	for i in range(num_frames):
		start = i * hop
		frame_T = x_padded[start:start + frame_len, :]
		next_start = (i + 1) * hop
		if next_start + frame_len <= x_padded.shape[0]:
			next_frame_T = x_padded[next_start:next_start + frame_len, :]
		else:
			next_frame_T = np.zeros_like(frame_T)

		# Determine frame type using SSC and apply filter bank to get MDCT coefficients.
		frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
		frame_F = filter_bank(frame_T, frame_type, win_type)

		if frame_type == "ESH":
			# Convert stacked 1024x2 into 128x8 per channel.
			chl = frame_F[:, 0].reshape(8, 128).T
			chr_ = frame_F[:, 1].reshape(8, 128).T
		else:
			chl = frame_F[:, 0].reshape(1024, 1)
			chr_ = frame_F[:, 1].reshape(1024, 1)

		# Apply TNS per channel.
		chl_tns, chl_coeffs = tns(chl, frame_type)
		chr_tns, chr_coeffs = tns(chr_, frame_type)

		# Append the frame dictionary to the AAC sequence list.
		aac_seq_2.append(
			{
				"frame_type": frame_type,
				"win_type": win_type,
				"chl": {"tns_coeffs": chl_coeffs, "frame_F": chl_tns},
				"chr": {"tns_coeffs": chr_coeffs, "frame_F": chr_tns},
			}
		)

		# Update previous frame type for next iteration.
		prev_frame_type = frame_type

	return aac_seq_2
