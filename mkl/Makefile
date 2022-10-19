MKLROOT=/opt/intel/oneapi/mkl/2022.1.0
MKLFLAGS= -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lgomp -lpthread -lm -ldl -m64  -I"${MKLROOT}/include"

# NOTE: $ source /opt/intel/oneapi/setvars.sh

all: compute write

compute: mkl.cpp mkl.hpp
	g++ $(MKLFLAGS) -o compute mkl.cpp

write: mkl.cpp mkl.hpp
	g++ $(MKLFLAGS) -DWRITE -o filewrite mkl.cpp

clean:
	rm compute