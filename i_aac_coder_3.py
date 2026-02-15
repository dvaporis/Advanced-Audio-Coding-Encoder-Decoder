import numpy as np
import soundfile as sf
from scipy.io import loadmat
import sys
import os

from i_filter_bank import i_filter_bank
from i_tns import i_tns
from i_AACquant import i_aac_quantizer

# Add the material directory to the path to import huff_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'material'))
from huff_utils import load_LUT, decode_huff


def i_aac_coder_3(aac_seq_3: list, filename_out: str) -> np.ndarray:
	"""
	Inverse AAC encoder level 3: decode Huffman, inverse quantization, inverse TNS, and reconstruct signal.
	
	Parameters
	----------
	aac_seq_3 : list
		List of AAC frame dictionaries from aac_coder_3
	filename_out : str
		Output audio file path
	
	Returns
	-------
	x : np.ndarray
		Reconstructed audio signal (stereo)
	"""
	if not isinstance(aac_seq_3, list):
		raise ValueError("aac_seq_3 must be a list")
	if len(aac_seq_3) == 0:
		raise ValueError("aac_seq_3 must not be empty")

	# Define frame length and hop size (overlap of 50%).
	frame_len = 2048
	hop = 1024
	num_frames = len(aac_seq_3)
	padded_len = num_frames * hop + 1024

	# Initialize padded array to hold reconstructed signal with overlap-add.
	x_padded = np.zeros((padded_len, 2))
	
	# Load Huffman codebooks
	huff_LUT = load_LUT()

	# Process each frame in the AAC sequence list.
	for i, frame in enumerate(aac_seq_3):
		frame_type = frame["frame_type"]
		win_type = frame.get("win_type", "KBD")

		# Extract channel data
		chl_data = frame["chl"]
		chr_data = frame["chr"]
		
		chl_tns_coeffs = chl_data["tns_coeffs"]
		chr_tns_coeffs = chr_data["tns_coeffs"]
		
		chl_G = chl_data["G"]
		chr_G = chr_data["G"]
		
		chl_sfc = chl_data["sfc"]  # Directly stored as integers
		chr_sfc = chr_data["sfc"]
		
		chl_stream = chl_data["stream"]
		chr_stream = chr_data["stream"]
		
		chl_codebook = chl_data["codebook"]
		chr_codebook = chr_data["codebook"]

		# Load table to get number of bands and reshape scale factors properly
		_dir = os.path.dirname(os.path.abspath(__file__))
		table_path = os.path.join(_dir, "TableB219.mat")
		table = loadmat(table_path)
		if frame_type == "ESH":
			bands = table["B219b"]
			n_bands = bands.shape[0]
			# Reshape to 42 bands x 8 subframes
			chl_sfc = np.array(chl_sfc, dtype=np.float64).reshape(n_bands, 8)
			chr_sfc = np.array(chr_sfc, dtype=np.float64).reshape(n_bands, 8)
		else:
			bands = table["B219a"]
			n_bands = bands.shape[0]
			# Reshape to n_bands x 1
			chl_sfc = np.array(chl_sfc, dtype=np.float64).reshape(n_bands, 1)
			chr_sfc = np.array(chr_sfc, dtype=np.float64).reshape(n_bands, 1)
		
		# Decode quantized coefficients S
		if chl_codebook == 0:
			chl_S = np.zeros(1024, dtype=np.int32)
		else:
			chl_S = decode_huff(chl_stream, huff_LUT[chl_codebook])
			chl_S = np.array(chl_S, dtype=np.int32)
			if len(chl_S) < 1024:
				chl_S = np.pad(chl_S, (0, 1024 - len(chl_S)), mode='constant')
			elif len(chl_S) > 1024:
				chl_S = chl_S[:1024]
		chl_S = chl_S.reshape(1024, 1)
		
		if chr_codebook == 0:
			chr_S = np.zeros(1024, dtype=np.int32)
		else:
			chr_S = decode_huff(chr_stream, huff_LUT[chr_codebook])
			chr_S = np.array(chr_S, dtype=np.int32)
			if len(chr_S) < 1024:
				chr_S = np.pad(chr_S, (0, 1024 - len(chr_S)), mode='constant')
			elif len(chr_S) > 1024:
				chr_S = chr_S[:1024]
		chr_S = chr_S.reshape(1024, 1)

		# Apply inverse quantization per channel
		chl_T_reconstructed = i_aac_quantizer(chl_S, chl_sfc, chl_G, frame_type)
		chr_T_reconstructed = i_aac_quantizer(chr_S, chr_sfc, chr_G, frame_type)

		# Apply inverse TNS per channel
		chl_F_reconstructed = i_tns(chl_T_reconstructed, frame_type, chl_tns_coeffs)
		chr_F_reconstructed = i_tns(chr_T_reconstructed, frame_type, chr_tns_coeffs)

		# Convert the frame_F back to the original shape based on frame type
		if frame_type == "ESH":
			if chl_F_reconstructed.shape != (128, 8) or chr_F_reconstructed.shape != (128, 8):
				raise ValueError("ESH frames must be 128x8 per channel")
			frame_F = np.zeros((1024, 2))
			frame_F[:, 0] = chl_F_reconstructed.T.reshape(1024)
			frame_F[:, 1] = chr_F_reconstructed.T.reshape(1024)
		else:
			if chl_F_reconstructed.shape[0] != 1024 or chr_F_reconstructed.shape[0] != 1024:
				raise ValueError("Non-ESH frames must be 1024x1 per channel")
			frame_F = np.zeros((1024, 2))
			frame_F[:, 0] = chl_F_reconstructed.reshape(1024)
			frame_F[:, 1] = chr_F_reconstructed.reshape(1024)

		# Apply the inverse filter bank to get the time-domain frame
		frame_T = i_filter_bank(frame_F, frame_type, win_type)

		start = i * hop
		x_padded[start:start + frame_len, :] += frame_T

	# Remove the initial and final 1024 zero padding added in the encoder
	x = x_padded[1024:-1024, :]

	# Normalize to prevent clipping and write as 16-bit PCM
	# Clip to [-1, 1] range for safety
	x = np.clip(x, -1.0, 1.0)
	
	# Write the reconstructed signal to the output file as 16-bit PCM
	sf.write(filename_out, x, 48000, subtype='PCM_16')
	return x
