NVCC = nvcc
NVCC_FLAGS = -arch=sm_75

TARGETS = vecAdd vecAddStream

all: $(TARGETS)

vecAdd: vecAdd.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

vecAddStream: vecAddStream.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
