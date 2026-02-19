import os
import numpy as np
import soundfile as sf


def compute_snr(reference: np.ndarray, reconstructed: np.ndarray) -> float:
	min_len = min(reference.shape[0], reconstructed.shape[0])
	ref = reference[:min_len, :].astype(np.float64)
	rec = reconstructed[:min_len, :].astype(np.float64)
	noise = ref - rec
	signal_power = np.sum(ref ** 2)
	noise_power = np.sum(noise ** 2)
	if noise_power == 0:
		return float("inf")
	return 10 * np.log10(signal_power / noise_power)


def main() -> None:
	x, fs = sf.read("LicorDeCalandraca.wav", always_2d=True)

	x1, _ = sf.read("output_1.wav", always_2d=True)
	x2, _ = sf.read("output_2.wav", always_2d=True)
	x3, _ = sf.read("output_3.wav", always_2d=True)

	snr1 = compute_snr(x, x1)
	snr2 = compute_snr(x, x2)
	snr3 = compute_snr(x, x3)

	duration = x.shape[0] / fs
	original_kbps = (os.path.getsize("LicorDeCalandraca.wav") * 8) / duration / 1000
	coded3_kbps = (os.path.getsize("aac_coded_3.mat") * 8) / duration / 1000
	compression = original_kbps / coded3_kbps

	print(f"SNR1_DB={snr1:.4f}")
	print(f"SNR2_DB={snr2:.4f}")
	print(f"SNR3_DB={snr3:.4f}")
	print(f"ORIG_KBPS_FILE={original_kbps:.4f}")
	print(f"CODED3_KBPS_FILE={coded3_kbps:.4f}")
	print(f"COMPRESSION3_FILE={compression:.4f}")


if __name__ == "__main__":
	main()