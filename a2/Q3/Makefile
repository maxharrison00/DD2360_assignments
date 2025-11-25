NVCC = nvcc
NVCC_FLAGS = -arch=sm_75

TARGETS = matrixMultiplicationShared

all: $(TARGETS)

$(TARGETS): $(TARGETS).cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
clean:
	rm -f $(TARGETS)