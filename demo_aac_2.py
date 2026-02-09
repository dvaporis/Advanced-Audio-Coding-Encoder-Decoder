import numpy as np
import soundfile as sf

from aac_coder_2 import aac_coder_2
from i_aac_coder_2 import i_aac_coder_2


def demo_aac_2(filename_in: str, filename_out: str) -> float:
	"""Run AAC coder level 2 and return overall SNR in dB."""
	# Encode
	aac_seq_2 = aac_coder_2(filename_in)

	# Decode and write output
	x_rec = i_aac_coder_2(aac_seq_2, filename_out)

	# Read original input for SNR calculation
	x, fs = sf.read(filename_in, always_2d=True)
	if fs != 48000:
		raise ValueError("Input must be 48 kHz")

	if not np.issubdtype(x.dtype, np.floating):
		x = x.astype(np.float64)
	else:
		x = x.astype(np.float64)

	# Align lengths
	min_len = min(x.shape[0], x_rec.shape[0])
	x = x[:min_len, :]
	x_rec = x_rec[:min_len, :]

	noise = x - x_rec
	signal_power = np.sum(x ** 2)
	noise_power = np.sum(noise ** 2)
	if noise_power == 0:
		return float("inf")

	snr = 10 * np.log10(signal_power / noise_power)
	return snr


# Run the demo with the specified input and output files, and print the SNR.
snr = demo_aac_2("LicorDeCalandraca.wav", "output_2.wav")
print(f"SNR: {snr:.2f} dB")
