NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS=-O3 -lcusparse -w

all: cusparse block

cusparse: cusparse.cu cusparse.hpp
	$(NVCC) $(NVCCFLAGS) -DCOMPUTE -o cusparse cusparse.cu

block: block_cusparse.cu cusparse.hpp
	$(NVCC) $(NVCCFLAGS) -DCOMPUTE -o block_cusparse block_cusparse.cu

compute: cusparse.cu cusparse.hpp
	$(NVCC) $(NVCCFLAGS) -DCOMPUTE -o compute cusparse.cu

d2h: cusparse.cu cusparse.hpp
	$(NVCC) $(NVCCFLAGS) -DD2H -o d2h cusparse.cu

write: cusparse.cu cusparse.hpp
	$(NVCC) $(NVCCFLAGS) -DD2H -DWRITE -o filewrite cusparse.cu

clean:
	rm compute d2h
