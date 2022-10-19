BASEDIR=/home/jiet/DBD/icde/large # Change this BASEDIR
SRCDIR=$(BASEDIR)/src
OBJDIR=$(BASEDIR)/obj

INCFLAGS=-I/usr/local/cuda/include/ -I./src/
LINKFLAGS=-L/usr/local/cuda/lib64 -L$(SRCDIR)/
CPP=g++
CPPFLAGS=-O3 -std=c++14 --no-warn -fopenmp -lpthread -w -Wno-strict-aliasing
CPPFLAGSDEBUG=$(CPPFLAGS) -DDEBUG
CPPFLAGSCOMPUTE=$(CPPFLAGS) -DCOMPUTE
NVCC=nvcc
NVCCFLAGS=-arch=compute_86 -code=sm_86 -Xcompiler -fopenmp -lpthread -O3 -w $(LINKFLGAS) # Change CC value
NVCCFLAGSDEBUG=$(NVCCFLAGS) -DDEBUG
NVCCFLAGSCOMPUTE=$(NVCCFLAGS) -DCOMPUTE
HEADER=$(SRCDIR)/SSpMM.hpp

all: SSpMM compute d2h


SSpMM: $(OBJDIR)/main.o $(OBJDIR)/readInput.o $(OBJDIR)/preProc.o $(OBJDIR)/distributeWorkload.o $(OBJDIR)/computeCPU.o $(OBJDIR)/computeGPU.o $(OBJDIR)/writeFile.o
	$(NVCC) $(NVCCFLAGS) -o bin/SSpMM $^

compute: $(OBJDIR)/main_compute.o $(OBJDIR)/readInput.o $(OBJDIR)/preProc_compute.o $(OBJDIR)/distributeWorkload_compute.o $(OBJDIR)/computeCPU_compute.o $(OBJDIR)/computeGPU_compute.o# $(HEADER)
	$(NVCC) $(NVCCFLAGSCOMPUTE) -o bin/compute $^

debug: $(OBJDIR)/main_debug.o $(OBJDIR)/readInput.o $(OBJDIR)/preProc_debug.o $(OBJDIR)/distributeWorkload_debug.o $(OBJDIR)/computeCPU_debug.o $(OBJDIR)/computeGPU_debug.o $(OBJDIR)/writeFile_debug.o
	$(NVCC) $(NVCCFLAGSDEBUG) -o bin/debug $^

d2h: $(OBJDIR)/main_compute.o $(OBJDIR)/readInput.o $(OBJDIR)/preProc_d2h.o $(OBJDIR)/distributeWorkload_compute.o $(OBJDIR)/computeCPU_compute.o $(OBJDIR)/computeGPU_d2h.o# $(HEADER)
	$(NVCC) $(NVCCFLAGSCOMPUTE) -DD2H -o bin/d2h $^


# End-to-End L3SpMM
$(OBJDIR)/preProc.o: $(SRCDIR)/preProc.cu $(HEADER)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<
	
$(OBJDIR)/computeGPU.o: $(SRCDIR)/computeGPU.cu $(HEADER)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<



# For Kernel Evaluation
$(OBJDIR)/preProc_compute.o: $(SRCDIR)/preProc.cu $(HEADER)
	$(NVCC) $(NVCCFLAGSCOMPUTE) -c -o $@ $<
	
$(OBJDIR)/computeGPU_compute.o: $(SRCDIR)/computeGPU.cu $(HEADER)
	$(NVCC) $(NVCCFLAGSCOMPUTE) -c -o $@ $<


# For Kernel + Device-to-Host Transfer Evaluation
$(OBJDIR)/computeGPU_d2h.o: $(SRCDIR)/computeGPU.cu $(HEADER)
	$(NVCC) $(NVCCFLAGSCOMPUTE) -DD2H -c -o $@ $<

$(OBJDIR)/preProc_d2h.o: $(SRCDIR)/preProc.cu $(HEADER)
	$(NVCC) $(NVCCFLAGSCOMPUTE) -DD2H -c -o $@ $<


# Debug
$(OBJDIR)/preProc_debug.o: $(SRCDIR)/preProc.cu $(HEADER)
	$(NVCC) $(NVCCFLAGSDEBUG) -c -o $@ $<
	
$(OBJDIR)/computeGPU_debug.o: $(SRCDIR)/computeGPU.cu $(HEADER)
	$(NVCC) $(NVCCFLAGSDEBUG) -c -o $@ $<

clean:
#	rm obj/*
	rm bin/*

