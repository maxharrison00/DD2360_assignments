import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os

array_lengths = [1024, 10240, 102400, 1024000]
distributions = ["Uniform", "Normal"]
cuda_executable = "./plotting_histogram"
output_file = "histogram_output.txt"

histogram_data = {}

for length in array_lengths:
    for dist in distributions:
        key = (length, dist)
        print(f"Running: Length={length}, Distribution={dist}\n")

        result = subprocess.run(
            [cuda_executable, str(length), dist],
            capture_output=True,
            text=True
        )

        print(result.stdout)

        bins = []
        counts = []

        with open(output_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    bins.append(int(parts[0]))
                    counts.append(int(parts[1]))

        histogram_data[key] = (np.array(bins), np.array(counts))

        for line in result.stdout.split('\n'):
            if 'Grid dimensions' in line or 'Threads per block' in line:
                print(f"  {line.strip()}")

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
fig.suptitle('Sampled Histograms', fontsize=16, fontweight='bold', y=0.995)

fig.text(0.28, 0.97, 'Uniform Distribution', ha='center', fontsize=13, fontweight='bold')
fig.text(0.73, 0.97, 'Normal Distribution', ha='center', fontsize=13, fontweight='bold')

for row_idx, length in enumerate(array_lengths):
    for col_idx, dist in enumerate(distributions):
        ax = axes[row_idx, col_idx]
        key = (length, dist)

        if key in histogram_data:
            bins, counts = histogram_data[key]

            ax.bar(bins, counts, width=max(1, len(bins)//1000),
                   edgecolor='none', alpha=0.7, color='steelblue' if dist == 'Uniform' else 'coral')

            if col_idx == 0:
                ax.set_ylabel(f'N = {length:,}\n\nCount', fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('Count', fontsize=10)

            if row_idx == len(array_lengths) - 1:
                ax.set_xlabel('Bin Number', fontsize=10)

            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            total_elements = np.sum(counts)
            non_zero_bins = np.sum(counts > 0)
            max_count = np.max(counts)

            stats_text = f'Total: {total_elements:,}\nNon-zero bins: {non_zero_bins}\nMax: {max_count}'
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6),
                   fontsize=8)

            ax.set_ylim(0, 135)

        else:
            ax.text(0.5, 0.5, 'Data not available',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_ylabel('Count', fontsize=10)
            if row_idx == len(array_lengths) - 1:
                ax.set_xlabel('Bin Number', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_filename = 'histogram_comparison.png'
plt.savefig(output_filename, dpi=200, bbox_inches='tight')

plt.show()
