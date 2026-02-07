import numpy as np
import soundfile as sf

from i_filter_bank import i_filter_bank


def i_aac_coder_1(aac_seq_1: list, filename_out: str) -> np.ndarray:
	"""Inverse AAC encoder level 1: reconstruct time signal and write wav."""
	if not isinstance(aac_seq_1, list):
		raise ValueError("aac_seq_1 must be a list")
	if len(aac_seq_1) == 0:
		raise ValueError("aac_seq_1 must not be empty")

    # Define frame length and hop size (overlap of 50%).
	frame_len = 2048
	hop = 1024
	num_frames = len(aac_seq_1)
	padded_len = num_frames * hop + 1024

    # Initialize padded array to hold reconstructed signal with overlap-add.
	x_padded = np.zeros((padded_len, 2))

    # Process each frame in the AAC sequence list and apply inverse filter bank to reconstruct time-domain signal.
	for i, frame in enumerate(aac_seq_1):
		frame_type = frame["frame_type"]
		win_type = frame["win_type"]

		chl = frame["chl"]["frame_F"]
		chr_ = frame["chr"]["frame_F"]

        # Convert the frame_F back to the original shape based on frame type.
		if frame_type == "ESH":
			if chl.shape != (128, 8) or chr_.shape != (128, 8):
				raise ValueError("ESH frames must be 128x8 per channel")
			frame_F = np.zeros((1024, 2))
			frame_F[:, 0] = chl.T.reshape(1024)
			frame_F[:, 1] = chr_.T.reshape(1024)
		else:
			if chl.shape[0] != 1024 or chr_.shape[0] != 1024:
				raise ValueError("Non-ESH frames must be 1024x1 per channel")
			frame_F = np.zeros((1024, 2))
			frame_F[:, 0] = chl.reshape(1024)
			frame_F[:, 1] = chr_.reshape(1024)

        # Apply the inverse filter bank to get the time-domain frame.
		frame_T = i_filter_bank(frame_F, frame_type, win_type)

		start = i * hop
		x_padded[start:start + frame_len, :] += frame_T

	# Remove the initial and final 1024 zero padding added in the encoder.
	x_rec = x_padded[1024:-1024, :]

    # Write the reconstructed signal to the output file.
	sf.write(filename_out, x_rec, 48000)
	return x_rec
