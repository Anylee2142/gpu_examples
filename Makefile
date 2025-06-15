# -----[ basic configuration ]-----------------------------------------------
CUDA_PATH ?= /usr/local/cuda
NVCC      ?= $(CUDA_PATH)/bin/nvcc

# Override on the command line if you like, e.g.:
#   make NVCCFLAGS="-O3 -arch=sm_80"
NVCCFLAGS ?= -O3

# ---------------------------------------------------------------------------
SRCS := $(wildcard *.cu)          # all .cu files in this dir
BINS := $(patsubst %.cu,%,$(SRCS))# strip extension → binary names

# Default target: build everything
all: $(BINS)

# Pattern rule: <binary> depends on matching .cu file
%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Clean up build products
.PHONY: clean
clean:
	rm -f $(BINS) *.o

# Convenience target for rebuilding from scratch
.PHONY: rebuild
rebuild: clean all

