import numpy as np
import soundfile as sf

from aac_coder_3 import aac_coder_3
from i_aac_coder_3 import i_aac_coder_3


def demo_aac_3(filename_in: str, filename_out: str, filename_aac_coded: str) -> tuple[float, float, float]:
	"""
	Run AAC coder level 3 and return SNR, bitrate, and compression ratio.
	
	Parameters
	----------
	filename_in : str
		Input WAV file (must be 48 kHz stereo).
	filename_out : str
		Output WAV file for reconstructed audio.
	filename_aac_coded : str
		.mat file to save the AAC encoded sequence.
	
	Returns
	-------
	SNR : float
		Signal-to-noise ratio in dB.
	bitrate : float
		Bitrate in kbps.
	compression : float
		Compression ratio (original_bitrate / compressed_bitrate).
	"""
	# Encode - aac_coder_3 now saves to filename_aac_coded itself
	aac_seq_3 = aac_coder_3(filename_in, filename_aac_coded)

	# Calculate total bitstream length
	# Count bits from scale factors (sfc), quantized coefficients (stream), and codebooks
	total_bits = 0
	for frame in aac_seq_3:
		# Left channel
		# Scale factors stored as integers (not Huffman encoded)
		chl_sfc_size = frame["chl"]["sfc"].size * 8  # 8 bits per integer
		total_bits += chl_sfc_size
		total_bits += len(frame["chl"]["stream"])  # Huffman encoded quantized coefficients
		
		# Right channel
		chr_sfc_size = frame["chr"]["sfc"].size * 8
		total_bits += chr_sfc_size
		total_bits += len(frame["chr"]["stream"])
		
		# Add overhead for codebook indices and metadata (approximate)
		total_bits += 16  # frame_type, win_type, codebook indices

	# Decode and write output
	x_rec = i_aac_coder_3(aac_seq_3, filename_out)

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

	# Calculate SNR
	noise = x - x_rec
	signal_power = np.sum(x ** 2)
	noise_power = np.sum(noise ** 2)
	if noise_power == 0:
		SNR = float("inf")
	else:
		SNR = 10 * np.log10(signal_power / noise_power)
	
	# Calculate bitrate and compression ratio
	duration = min_len / fs  # seconds
	original_bitrate = 2 * 16 * fs / 1000  # 2 channels * 16 bits * 48000 Hz -> kbps
	bitrate = total_bits / duration / 1000  # kbps (compressed AAC bitstream)
	compression = original_bitrate / bitrate
	
	print(f"SNR: {SNR:.2f} dB")
	print(f"Total bits: {total_bits}")
	print(f"Duration: {duration:.2f} seconds")
	print(f"Original PCM bitrate: {original_bitrate:.2f} kbps")
	print(f"AAC compressed bitrate: {bitrate:.2f} kbps")
	print(f"Compression ratio: {compression:.2f}:1")
	print(f"\nNote: Output file '{filename_out}' is uncompressed PCM at {original_bitrate:.2f} kbps")
	
	return SNR, bitrate, compression


if __name__ == "__main__":
	# Example usage
	print("=" * 60)
	print("Testing AAC Level 3 Encoder")
	print("=" * 60)
	SNR, bitrate, compression = demo_aac_3(
		"LicorDeCalandraca.wav", 
		"output_3.wav",
		"aac_coded_3.mat"
	)
	print(f"\nFinal Results:")
	print(f"  SNR: {SNR:.2f} dB")
	print(f"  Bitrate: {bitrate:.2f} kbps")
	print(f"  Compression: {compression:.2f}:1")
