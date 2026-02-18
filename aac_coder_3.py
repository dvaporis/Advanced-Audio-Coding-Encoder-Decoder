import numpy as np
import soundfile as sf
from scipy.io import savemat
import sys
import os

from SSC import SSC
from filter_bank import filter_bank
from tns import tns
from psycho import psycho
from aac_quantizer import aac_quantizer

# Add the material directory to the path to import huff_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'material'))
from huff_utils import load_LUT, encode_huff


def aac_coder_3(filename_in: str, filename_aac_coded: str) -> list:
	"""
	AAC encoder level 3: apply psychoacoustic model, TNS, quantization, and Huffman encoding.
	
	Parameters
	----------
	filename_in : str
		Input audio file path (must be 48 kHz stereo)
	filename_aac_coded : str
		Output .mat file to save the aac_seq_3 structure
	
	Returns
	-------
	aac_seq_3 : list
		List of AAC frame dictionaries containing:
		- frame_type : str
		- win_type : str (for ESH frames)
		- chl : dict with {tns_coeffs, T, G, sfc, stream, codebook}
		- chr : dict with {tns_coeffs, T, G, sfc, stream, codebook}
	"""
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
	aac_seq_3 = []
	prev_frame_type = "OLS"
	win_type = "KBD"
	
	# Load Huffman codebooks
	huff_LUT = load_LUT()
	
	# Store previous frames for psychoacoustic model
	prev_frame_T_1 = None
	prev_frame_T_2 = None

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
			chl_F = frame_F[:, 0].reshape(8, 128).T
			chr_F = frame_F[:, 1].reshape(8, 128).T
		else:
			chl_F = frame_F[:, 0].reshape(1024, 1)
			chr_F = frame_F[:, 1].reshape(1024, 1)

		# Apply psychoacoustic model per channel to get SMR
		chl_SMR = psycho(frame_T[:, 0], frame_type,
						 prev_frame_T_1[:, 0] if prev_frame_T_1 is not None else None,
						 prev_frame_T_2[:, 0] if prev_frame_T_2 is not None else None)
		chr_SMR = psycho(frame_T[:, 1], frame_type,
						 prev_frame_T_1[:, 1] if prev_frame_T_1 is not None else None,
						 prev_frame_T_2[:, 1] if prev_frame_T_2 is not None else None)

		# Apply TNS per channel.
		chl_T, chl_tns_coeffs = tns(chl_F, frame_type)
		chr_T, chr_tns_coeffs = tns(chr_F, frame_type)

		# Apply quantization per channel - returns (S, sfc, G)
		chl_S, chl_sfc, chl_G = aac_quantizer(chl_T, frame_type, chl_SMR)
		chr_S, chr_sfc, chr_G = aac_quantizer(chr_T, frame_type, chr_SMR)

		# Store scale factors directly (not Huffman encoded due to range limitations)
		chl_sfc_int = np.round(chl_sfc).astype(np.int32)
		chr_sfc_int = np.round(chr_sfc).astype(np.int32)
		
		# Huffman encode the quantized MDCT coefficients (S)
		chl_S_flat = chl_S.flatten()
		chl_stream, chl_codebook = encode_huff(chl_S_flat, huff_LUT)
		
		chr_S_flat = chr_S.flatten()
		chr_stream, chr_codebook = encode_huff(chr_S_flat, huff_LUT)

		# Build the frame dictionary according to spec
		frame_dict = {
			"frame_type": frame_type,
			"chl": {
				"tns_coeffs": chl_tns_coeffs,
				"T": chl_T,
				"G": chl_G,
				"sfc": chl_sfc_int,  # Scale factors as integers
				"stream": chl_stream,    # Huffman encoded quantized coefficients
				"codebook": chl_codebook,
			},
			"chr": {
				"tns_coeffs": chr_tns_coeffs,
				"T": chr_T,
				"G": chr_G,
				"sfc": chr_sfc_int,
				"stream": chr_stream,
				"codebook": chr_codebook,
			},
		}
		
		# Add win_type for ESH frames
		if frame_type == "ESH":
			frame_dict["win_type"] = win_type
		
		aac_seq_3.append(frame_dict)

		# Update previous frames for psychoacoustic model
		prev_frame_T_2 = prev_frame_T_1
		prev_frame_T_1 = frame_T
		prev_frame_type = frame_type

	# Save aac_seq_3 to .mat file as per specification
	savemat(filename_aac_coded, {"aac_seq_3": aac_seq_3})

	return aac_seq_3
