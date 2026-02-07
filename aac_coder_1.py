import numpy as np
import soundfile as sf

from SSC import SSC
from filter_bank import filter_bank


def aac_coder_1(filename_in: str) -> list:
	"""AAC encoder level 1: returns list of frame dictionaries with MDCT coefficients."""
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

	frame_len = 2048
	hop = 1024
	num_frames = (x_padded.shape[0] - frame_len) // hop + 1

	aac_seq_1 = []
	prev_frame_type = "OLS"
	win_type = "KBD"

	for i in range(num_frames):
		start = i * hop
		frame_T = x_padded[start:start + frame_len, :]
		next_start = (i + 1) * hop
		if next_start + frame_len <= x_padded.shape[0]:
			next_frame_T = x_padded[next_start:next_start + frame_len, :]
		else:
			next_frame_T = np.zeros_like(frame_T)

		frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
		frame_F = filter_bank(frame_T, frame_type, win_type)

		if frame_type == "ESH":
			# Convert stacked 1024x2 into 128x8 per channel.
			chl = frame_F[:, 0].reshape(8, 128).T
			chr_ = frame_F[:, 1].reshape(8, 128).T
		else:
			chl = frame_F[:, 0].reshape(1024, 1)
			chr_ = frame_F[:, 1].reshape(1024, 1)

		aac_seq_1.append(
			{
				"frame_type": frame_type,
				"win_type": win_type,
				"chl": {"frame_F": chl},
				"chr": {"frame_F": chr_},
			}
		)

		prev_frame_type = frame_type

	return aac_seq_1
