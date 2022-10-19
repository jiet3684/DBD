#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define BLOCKSIZE 512
#define ITER 5

struct edgeData {
    int from;
    int to;
    float val;
};
typedef struct edgeData edgeData;

struct csr {
    int *ptr;
    int *idx;
    float *val;
    int nr;
    int nc;
    int ne;
};
typedef struct csr csr;
