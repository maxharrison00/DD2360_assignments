NVCC = nvcc
NVCC_FLAGS = -arch=sm_75

TARGETS = vecAdd matrixMultiplication

all: $(TARGETS)

vecAdd: Q2/vecAdd.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

matrixMultiplication: Q3/matrixMultiplication.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
