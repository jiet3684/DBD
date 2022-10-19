#compilers
CC=nvcc

#includes
CUDA_INSTALL_PATH = /usr/local/cuda
LIBRARY_PATH=/home/jiet/Documents/Library # Modify This
CUSP_PATH=$(LIBRARY_PATH)/cusplibrary-0.5.1
THRUST_PATH=$(LIBRARY_PATH)/Thrust-master
HIP_PATH=$(LIBRARY_PATH)/hcc2-hip/include
HELPER_PATH=$(LIBRARY_PATH)/cuda-samples/Common
INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc -I$(CUSP_PATH) -I$(THRUST_PATH) -I$(HIP_PATH) -I$(HELPER_PATH)

#libs
#CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart
#LIBS = $(CUDA_LIBS)

#-gencode=arch=compute_35,code=sm_35
#-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52

all: bhsparse block

bhsparse: main.cu
	nvcc -O3 -m64  -gencode=arch=compute_86,code=compute_86 main.cu -o bhsparse $(INCLUDES) $(LIBS)

block: block_main.cu
	nvcc -O3 -m64  -gencode=arch=compute_86,code=compute_86 block_main.cu -o block_bhsparse $(INCLUDES) $(LIBS)

