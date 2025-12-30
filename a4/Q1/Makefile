NVCC = nvcc
NVCC_FLAGS = -arch=sm_75

TARGETS = convolution

all: $(TARGETS)

hw4-convolution: convolution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGETS)