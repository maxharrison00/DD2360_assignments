import subprocess
import matplotlib.pyplot as plt
import numpy as np

START_SIZE = 512
NUM_RUNS = 10
EXECUTABLE = "./opt_reduction"

array_sizes = []
cpu_times = []
gpu_times = []

for i in range(NUM_RUNS):
    array_size = START_SIZE * (2 ** i)
    array_sizes.append(array_size)

    try:
        result = subprocess.run(
            [EXECUTABLE, str(array_size)],
            capture_output=True,
            text=True,
            check=True
        )

        output_lines = result.stdout.split('\n')
        cpu_time = None
        gpu_time = None

        for line in output_lines:
            if "CPU time:" in line:
                cpu_time = float(line.split(':')[1].strip().split()[0])
            elif "GPU time:" in line:
                gpu_time = float(line.split(':')[1].strip().split()[0])

        if cpu_time is not None and gpu_time is not None:
            cpu_times.append(cpu_time)
            gpu_times.append(gpu_time)
        else:
            cpu_times.append(0)
            gpu_times.append(0)

    except subprocess.CalledProcessError as e:
        cpu_times.append(0)
        gpu_times.append(0)
    except FileNotFoundError:
        break

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(array_sizes))
width = 0.35

bars1 = ax.bar(x - width/2, cpu_times, width, label='CPU', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, gpu_times, width, label='GPU', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Array Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
ax.set_title('Reduction: CPU vs GPU Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{size:,}' for size in array_sizes], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)

add_value_labels(bars1)
add_value_labels(bars2)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
