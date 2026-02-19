import subprocess
import matplotlib.pyplot as plt
import sys

# Run demo_aac_3.py with COMPRESSION_BIAS for different values
compression_values = range(0, 60, 5)
compression_rates = []
snr_values = []

for value in compression_values:
    result = subprocess.run(
        [sys.executable, 'demo_aac_3.py', str(value)],
        capture_output=True,
        text=True
    )
    # Parse compression rate and SNR from output
    rate = None
    snr = None
    for line in result.stdout.split('\n'):
        if 'Compression ratio:' in line:
            rate = float(line.split(':')[1].strip().split(':')[0])
        if line.startswith('SNR:'):
            snr = float(line.split(':', 1)[1].strip().split()[0])

    if rate is None or snr is None:
        raise RuntimeError(
            f"Failed to parse compression metrics for COMPRESSION_BIAS={value}.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    print(f"COMPRESSION_BIAS={value} -> Compression Ratio: {rate:.2f}:1, SNR: {snr:.2f} dB")
    compression_rates.append(rate)
    snr_values.append(snr)

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(list(compression_values), compression_rates, s=50, alpha=0.6)
plt.xlabel('COMPRESSION_BIAS')
plt.ylabel('Compression Rate')
plt.title('Compression Rates vs COMPRESSION_BIAS')
plt.grid(True, alpha=0.3)
plt.savefig('compression_rates_vs_bias.pdf', format='pdf', bbox_inches='tight')

# Plot SNR scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(list(compression_values), snr_values, s=50, alpha=0.6)
plt.xlabel('COMPRESSION_BIAS')
plt.ylabel('SNR (dB)')
plt.title('SNR vs COMPRESSION_BIAS')
plt.grid(True, alpha=0.3)
plt.savefig('snr_vs_bias.pdf', format='pdf', bbox_inches='tight')