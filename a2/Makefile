NVCC = nvcc
NVCC_FLAGS = -arch=sm_75

TARGETS = histogram reduction matrixMultiplicationShared

all: $(TARGETS)

histogram: Q1/histogram.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

reduction: Q2/reduction.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	
matrixMultiplicationShared: Q3/matrixMultiplicationShared.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
