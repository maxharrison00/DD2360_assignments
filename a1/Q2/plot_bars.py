import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('results.csv')

# Create stacked bar chart with log scale
plt.figure(figsize=(12, 8))

bars = df['ArrayLengthLog']
bar_positions = np.arange(len(bars))

# Plot stacked bars
plt.bar(bar_positions, df['HtoD_Time'], label='HtoD Copy', color='skyblue', edgecolor='black')
plt.bar(bar_positions, df['Kernel_Time'], bottom=df['HtoD_Time'], label='Kernel', color='lightcoral', edgecolor='black')
plt.bar(bar_positions, df['DtoH_Time'], bottom=df['HtoD_Time'] + df['Kernel_Time'], label='DtoH Copy', color='lightgreen', edgecolor='black')

# Use log scale
plt.yscale('log')

plt.xlabel('Array Length (log_2)', fontsize=12)
plt.ylabel('Time (us)', fontsize=12)
plt.title('CUDA Operation Timings - Stacked Bar Chart (Log Scale)', fontsize=14)
plt.legend()

plt.xticks(bar_positions, bars, rotation=45)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

plt.savefig('cuda_timings_stacked.png', dpi=300, bbox_inches='tight')
