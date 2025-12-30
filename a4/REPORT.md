# Assignment 4: Programming Productivity

- Assignment Group 3
- Giovanni Prete and Max Harrison

Both had equal contributions.

## Question 1 -  1D Convolution
### 1) Application Domains

Three primary domains where convolution is fundamentally applied include:
1.  **Digital Signal Processing (DSP):** Used for 1D signal filtering, such as noise reduction, equalization, and echo cancellation in audio processing.
2.  **Image Processing & Computer Vision:** Fundamental for operations like edge detection (e.g., Sobel filter), blurring (Gaussian blur), and sharpening images.
3.  **Deep Learning:** The core operation in Convolutional Neural Networks (CNNs) for feature extraction in tasks like image classification, object detection, and natural language processing.

### 2) Tiled Kernel Design and Tile Size

To design the tiled kernel, we utilize Shared Memory to reduce redundant global memory accesses. The main challenge in convolution is the "halo" effect: computing an output pixel requires input elements that might belong to the neighboring block's territory.


Each thread block loads a portion of the input array into shared memory. Since the mask width is 5 (radius 2), we need to load the "internal" pixels corresponding to the block's threads, plus a "halo" of 2 pixels on the left and 2 pixels on the right. We align the output tile size with the block dimension (`blockDim.x`). Threads near the boundaries of the block are responsible for loading the extra halo elements from global memory into the shared memory buffer. After a `__syncthreads()` barrier, all threads can compute the convolution reading solely from the fast shared memory.

The tile size is typically chosen to match the block size or a multiple of the warp size (32) to maximize occupancy. A common choice is 256 or 512 threads. Larger tiles are generally better as they minimize the ratio of "halo" nodes to "internal" nodes (overhead), reducing the relative cost of loading the extra boundary pixels. However, the size is limited by the amount of available shared memory per Streaming Multiprocessor (SM).

### 3) Memory Reads/Writes Analysis

**Basic Implementation:**
* **Global Reads:** For each output element $i$, the kernel iterates through the mask of size $M=5$. Thus, it performs 5 global reads per output. For an array of size $N$, total reads = $5N$.
* **Global Writes:** Each thread writes exactly 1 output. Total writes = $1N$.

**Tiled Implementation:**
* **Global Reads:** Data is loaded into shared memory once per block. For a tile of size $T$ (where $T$ outputs are computed), the block loads $T$ internal elements + 4 halo elements.
    * Reads per output $\approx (T + 4) / T = 1 + 4/T$.
    * For large $T$ (e.g., 256), this approaches 1 global read per output. Total reads $\approx N (1 + 4/T)$.
* **Global Writes:** Same as basic, 1 global write per output. Total writes = $1N$.

**Comparison:** The tiled version reduces global memory read traffic by a factor of approximately 5x.

### 4) Performance Benchmark

The following benchmark runs the program with input sizes $N$ of 1024, 2048, 4096, 8192, and 16384.

```text
================================================================
 RUNNING TEST CASE: Input Size N = 1024
================================================================
Timing - Allocated host memory. 		Elasped 6 microseconds 
Timing - Allocated device memory. 		Elasped 215007 microseconds 
Timing - Finished 1D convolution (CPU Reference). 		Elasped 36 microseconds 
Timing - Copying data to the GPU.. 		Elasped 272 microseconds 
Timing - Finished 1D convolution(basic). 		Elasped 146 microseconds 
Timing - Finished 1D convolution(tiled). 		Elasped 26 microseconds 
Timing - Copying output P to the CPU. 		Elasped 20 microseconds 
Validation (GPU vs CPU): Total Error = 0.000000
TEST PASSED

================================================================
 RUNNING TEST CASE: Input Size N = 2048
================================================================
Timing - Allocated host memory. 		Elasped 7 microseconds 
Timing - Allocated device memory. 		Elasped 200599 microseconds 
Timing - Finished 1D convolution (CPU Reference). 		Elasped 69 microseconds 
Timing - Copying data to the GPU.. 		Elasped 1198 microseconds 
Timing - Finished 1D convolution(basic). 		Elasped 133 microseconds 
Timing - Finished 1D convolution(tiled). 		Elasped 30 microseconds 
Timing - Copying output P to the CPU. 		Elasped 34 microseconds 
Validation (GPU vs CPU): Total Error = 0.000000
TEST PASSED

================================================================
 RUNNING TEST CASE: Input Size N = 4096
================================================================
Timing - Allocated host memory. 		Elasped 6 microseconds 
Timing - Allocated device memory. 		Elasped 203557 microseconds 
Timing - Skipped CPU Reference for large N. 		Elasped 0 microseconds 
Timing - Copying data to the GPU.. 		Elasped 275 microseconds 
Timing - Finished 1D convolution(basic). 		Elasped 142 microseconds 
Timing - Finished 1D convolution(tiled). 		Elasped 29 microseconds 
Timing - Copying output P to the CPU. 		Elasped 33 microseconds 
Validation skipped for large N

================================================================
 RUNNING TEST CASE: Input Size N = 8192
================================================================
Timing - Allocated host memory. 		Elasped 6 microseconds 
Timing - Allocated device memory. 		Elasped 201763 microseconds 
Timing - Skipped CPU Reference for large N. 		Elasped 0 microseconds 
Timing - Copying data to the GPU.. 		Elasped 916 microseconds 
Timing - Finished 1D convolution(basic). 		Elasped 153 microseconds 
Timing - Finished 1D convolution(tiled). 		Elasped 29 microseconds 
Timing - Copying output P to the CPU. 		Elasped 48 microseconds 
Validation skipped for large N

================================================================
 RUNNING TEST CASE: Input Size N = 16384
================================================================
Timing - Allocated host memory. 		Elasped 9 microseconds 
Timing - Allocated device memory. 		Elasped 200674 microseconds 
Timing - Skipped CPU Reference for large N. 		Elasped 0 microseconds 
Timing - Copying data to the GPU.. 		Elasped 270 microseconds 
Timing - Finished 1D convolution(basic). 		Elasped 147 microseconds 
Timing - Finished 1D convolution(tiled). 		Elasped 29 microseconds 
Timing - Copying output P to the CPU. 		Elasped 77 microseconds 
Validation skipped for large N

```

### 5) Profiling Results (N = 16384)
The analysis focuses on the optimized `convolution_1D_tiled` kernel.

* **Shared Memory Usage:** `1.04 Kbyte/block`
* **Achieved Occupancy:** `39.20 %`

**Analysis:**
1.  The usage of 1.04 KB perfectly matches the calculation. For a tile width of 256 and a mask size of 5 (radius 2), the kernel requires:
    $$(256 \text{ internal} + 4 \text{ halo}) \times 4 \text{ bytes/float} = 260 \times 4 = 1040 \text{ bytes} \approx 1.02 \text{ KB}$$
    This confirms that the tiled implementation correctly allocates only the necessary memory for the tile and its halo.

2.  **Occupancy:** The achieved occupancy is 39.20%. Although the kernel has low register and shared memory pressure (which theoretically allows for 100% occupancy), the achieved value is limited by the workload size.
    With $N=16,384$ and a block size of 256, the grid consists of only $\lceil 16384/256 \rceil = 64$ blocks. A Tesla T4 GPU has 40 Streaming Multiprocessors (SMs). Distributing 64 blocks across 40 SMs results in only 1.6 blocks per SM on average. Since the hardware can support many more active blocks per SM, the GPU is not fully saturated, leading to the observed occupancy of ~39%. To achieve higher occupancy, a significantly larger $N$ would be required to fill all SMs.

## Question 2 - NVIDIA Libraries and Managed Memory

### 1)

To approximate the FLOPS achieved in computing the SMPV, we can find the theoretical required number of floating-point operations required to compute the SMPV and then find the approximate time spent executing the SMPV. If we divide the theoretical number of FLOP's by the approximate time we can find the approximate FLOPS achieved during execution. This will not be exact due to the inaccuracy of time measurement and the likely more required FLOP's required by the actual cuSPARSE implementation.

By the definition of the diffusion matrix $A$ and the time step vector `tmp`, we can see that to compute the SMPV for a given time step we will need to perform 4 operations per element, for $dimX-2$ elements (as the first and last row in `tmp` are the constant 0). Thus for the entire computation we will need to perform $nsteps \cdot 4(dimX-2)$ FLOP's.
To approximate the time taken to compute the SMPV, we can utilise CUDA events to take time measurements before and after evaluation. Aggregating the individual time intervals over the time steps then results in a total execution time for computing the SMPV over the entire computation.

| dimX | SPMV time (s) | FLOPS (op/s) |
|------|---------------|--------------|
| 1024 | 0.0261 | 156,676,375.90 |
| 2048 | 0.0247 | 331,617,974.80 |
| 4096 | 0.0253 | 64,668,4831.97 |
| 8192 | 0.0235 | 1,393,746,011.49 |
| 16384 | 0.0299 | 2,192,158,437.04 |
| 32768 | 0.0252 | 5,210,670,695.34 |
| 65536 | 0.0334 | 7,849,793,376.06 |
| 131072 | 0.0486 | 10,791,429,100.71 |
| 262144 | 0.0957 | 10,953,504,162.79 |
| 524288 | 0.1734 | 12,095,720,935.06 |

As dimX increases, the measured SPMV time increases as expected due to the larger workload. In addition, the approximate FLOPS increases: this is likely due to the decreased ratio of overhead of calling the various kernels involved in the cuSPARSE implementation.

### 2)

Running the program 100 times with dimX=1024 for nsteps from 100 to 10000, we can find how the relative error of the approximation changes as the number of time steps increases in the following plot:

![Q2.2: Relative error for various timesteps](Q2/relative_error.png "immagine")

We can see from the plot that the relative error seems to converge as the number of time steps increases. Initially there are large decreases in relative error, but these decreases are smaller for the later time steps. This aligns with our intuition of a diffusion system as the system converges to a stable solution.

### 3)

The below table shows the average execution time over 20 iterations for various dimX sizes, with and without prefetching.

| dimX | Without prefetching (s) | With prefetching (s) |
|------|-------------------------|----------------------|
| 64 | 0.452827 | 0.441194 |
| 128 | 0.470727 | 0.383741 |
| 256 | 0.472843 | 0.403734 |
| 512 | 0.481950 | 0.452170 |
| 1024 | 0.531924 | 0.427101 |
| 2048 | 0.493554 | 0.427965 |
| 4096 | 0.516902 | 0.454684 |
| 8192 | 0.506130 | 0.422726 |
| 16384 | 0.490952 | 0.447741 |
| 32768 | 0.537801 | 0.438596 |

There seems to be a slight decrease in execution time with prefetching, with this difference being more pronounced for larger input values. This aligns with our intuition that as the required data for computation increases in size, the penalty of having to move data across devices increases as well.
