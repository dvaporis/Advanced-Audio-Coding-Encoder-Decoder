import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from SSC import SSC


def load_audio(path: str) -> tuple[np.ndarray, int]:
	x, fs = sf.read(path, always_2d=True)
	return x.astype(np.float64), fs


def compute_frame_types(stereo_signal: np.ndarray) -> list[str]:
	pad = np.zeros((1024, 2))
	x_padded = np.vstack([pad, stereo_signal, pad])

	frame_len = 2048
	hop = 1024
	num_frames = (x_padded.shape[0] - frame_len) // hop + 1

	frame_types: list[str] = []
	prev_frame_type = "OLS"

	for i in range(num_frames):
		start = i * hop
		frame_t = x_padded[start:start + frame_len, :]
		next_start = (i + 1) * hop
		if next_start + frame_len <= x_padded.shape[0]:
			next_frame_t = x_padded[next_start:next_start + frame_len, :]
		else:
			next_frame_t = np.zeros_like(frame_t)

		frame_type = SSC(frame_t, next_frame_t, prev_frame_type)
		frame_types.append(frame_type)
		prev_frame_type = frame_type

	return frame_types


def save_waveform_zoom(original: np.ndarray, out1: np.ndarray, out2: np.ndarray, out3: np.ndarray, fs: int, output_dir: str) -> None:
	start_sec = 0.0
	zoom_ms = 80.0
	start_idx = int(start_sec * fs)
	end_idx = min(start_idx + int((zoom_ms / 1000.0) * fs), len(original))
	if start_idx >= end_idx:
		start_idx = 0
		end_idx = min(int(0.08 * fs), len(original))

	time = np.arange(start_idx, end_idx) / fs

	fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
	series = [
		("Original", original),
		("Reconstructed L1", out1),
		("Reconstructed L2", out2),
		("Reconstructed L3", out3),
	]

	for ax, (title, sig) in zip(axes, series):
		end_idx_sig = min(end_idx, len(sig))
		time_sig = np.arange(start_idx, end_idx_sig) / fs
		ax.plot(time_sig, sig[start_idx:end_idx_sig, 0], linewidth=0.8)
		ax.set_ylabel("Amplitude")
		ax.set_title(title)
		ax.grid(alpha=0.25)

	axes[-1].set_xlabel("Time (s)")
	fig.suptitle("Waveform Zoom Comparison (Left Channel)")
	fig.tight_layout()
	fig.savefig(os.path.join(output_dir, "waveform_zoom_comparison.pdf"), format="pdf")
	plt.close(fig)


def save_error_signal(original: np.ndarray, reconstructed: np.ndarray, fs: int, output_dir: str) -> None:
	min_len = min(len(original), len(reconstructed))
	error = original[:min_len, 0] - reconstructed[:min_len, 0]
	time = np.arange(min_len) / fs

	fig, ax = plt.subplots(figsize=(11, 4))
	ax.plot(time, error, linewidth=0.6)
	ax.set_title("Error Signal Over Time (Original - Reconstructed L3, Left Channel)")
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Error Amplitude")
	ax.grid(alpha=0.3)
	fig.tight_layout()
	fig.savefig(os.path.join(output_dir, "error_signal_over_time.pdf"), format="pdf")
	plt.close(fig)


def db_spectrogram(signal_1d: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	freqs, times, spec = spectrogram(
		signal_1d,
		fs=fs,
		window="hann",
		nperseg=1024,
		noverlap=768,
		nfft=2048,
		mode="magnitude",
	)
	spec_db = 20.0 * np.log10(spec + 1e-10)
	return freqs, times, spec_db


def save_spectrogram_comparison(original: np.ndarray, out3: np.ndarray, fs: int, output_dir: str) -> None:
	min_len = min(len(original), len(out3))
	orig = original[:min_len, 0]
	rec = out3[:min_len, 0]

	f_o, t_o, s_o = db_spectrogram(orig, fs)
	f_r, t_r, s_r = db_spectrogram(rec, fs)

	vmin = min(np.percentile(s_o, 5), np.percentile(s_r, 5))
	vmax = max(np.percentile(s_o, 99), np.percentile(s_r, 99))

	fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)

	im0 = axes[0].pcolormesh(t_o, f_o, s_o, shading="gouraud", vmin=vmin, vmax=vmax)
	axes[0].set_title("Original Spectrogram (Left Channel)")
	axes[0].set_ylabel("Frequency (Hz)")

	axes[1].pcolormesh(t_r, f_r, s_r, shading="gouraud", vmin=vmin, vmax=vmax)
	axes[1].set_title("Reconstructed L3 Spectrogram (Left Channel)")
	axes[1].set_xlabel("Time (s)")
	axes[1].set_ylabel("Frequency (Hz)")

	cbar = fig.colorbar(im0, ax=axes, shrink=0.95)
	cbar.set_label("Magnitude (dB)")

	for ax in axes:
		ax.set_ylim(0, fs / 2)

	# fig.tight_layout()
	fig.savefig(os.path.join(output_dir, "spectrogram_comparison.png"), format="png", dpi=300)
	plt.close(fig)


def save_frame_type_distribution(frame_types: list[str], output_dir: str) -> None:
	labels = ["OLS", "LSS", "ESH", "LPS"]
	counts = [frame_types.count(label) for label in labels]
	total = sum(counts)
	percentages = [(count / total * 100.0) if total > 0 else 0.0 for count in counts]

	fig, ax = plt.subplots(figsize=(8, 5))
	bars = ax.bar(labels, counts)
	ax.set_title("Frame Type Distribution")
	ax.set_xlabel("Frame Type")
	ax.set_ylabel("Count")
	ax.grid(axis="y", alpha=0.25)

	for bar, pct in zip(bars, percentages):
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width() / 2, height, f"{pct:.1f}%", ha="center", va="bottom")

	fig.tight_layout()
	fig.savefig(os.path.join(output_dir, "frame_type_distribution.pdf"), format="pdf")
	plt.close(fig)


def main() -> None:
	output_dir = "figures"
	os.makedirs(output_dir, exist_ok=True)

    
	original, fs = load_audio("LicorDeCalandraca.wav")
	out1, fs1 = load_audio("output_1.wav")
	out2, fs2 = load_audio("output_2.wav")
	out3, fs3 = load_audio("output_3.wav")

	if fs != fs1 or fs != fs2 or fs != fs3:
		raise ValueError("Sampling rates do not match among input and output files.")

	frame_types = compute_frame_types(original)

	save_waveform_zoom(original, out1, out2, out3, fs, output_dir)
	save_error_signal(original, out3, fs, output_dir)
	save_spectrogram_comparison(original, out3, fs, output_dir)
	save_frame_type_distribution(frame_types, output_dir)

	print("Saved PDFs in figures/")
	print("- waveform_zoom_comparison.pdf")
	print("- error_signal_over_time.pdf")
	print("- spectrogram_comparison.pdf")
	print("- frame_type_distribution.pdf")


if __name__ == "__main__":
	main()