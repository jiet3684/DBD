#include "/opt/intel/oneapi/mkl/2022.1.0/include/mkl_spblas.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <string>
#include <unistd.h>
//#include <stdbool.h>

#define BLOCKSIZE 256
#define ITER 3

struct edgeData {
    int from;
    int to;
    float val;
};
typedef struct edgeData edgeData;

struct CSR {
    MKL_INT *ptr;
    MKL_INT *idx;
    float *val;
    MKL_INT nr;
    MKL_INT nc;
    MKL_INT ne;
};
typedef struct CSR CSR;
