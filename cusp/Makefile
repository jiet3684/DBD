#compilers
CC=nvcc

#includes
CUDA_INSTALL_PATH = /usr/local/cuda
LIBRARY_PATH=/home/jiet/Documents/Library # Modify This
CUSP_PATH=$(LIBRARY_PATH)/cusplibrary
THRUST_PATH=$(LIBRARY_PATH)/Thrust-master
#HIP_PATH=/home/jiet/Documents/Library/hipCPU/include/hipCPU/
HIP_PATH=$(LIBRARY_PATH)/hcc2-hip/include
INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc -I$(CUSP_PATH) -I$(CUSP_PATH)/io -I$(THRUST_PATH) -I$(HIP_PATH)

#libs
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse
LIBS = $(CUDA_LIBS)

#MKL
MKL_INSTALL_DIR = /opt/intel/mkl
MKL_INCLUDE = -I$(MKL_INSTALL_DIR)/include -I$(MKL_INSTALL_DIR)
MKL_LIBS = -L$(MKL_INSTALL_DIR)/lib/intel64 -lmkl_rt -ldl
#-gencode=arch=compute_35,code=sm_35
#-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52

all: cusp block

cusp: cusp.cu
	nvcc -O3 -m64 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 cusp.cu -o cusp $(INCLUDES) $(LIBS)

block: block_cusp.cu
	nvcc -O3 -m64 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 block_cusp.cu -o block_cusp $(INCLUDES) $(LIBS)
#	nvcc -O3 -m64 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52 spmm_hybrid.cu -o mkl $(INCLUDES) $(LIBS)


