#include <stdio.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <string>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>

#ifdef COMPUTE
    #define ROWS_IN_BLOCK 512
    #define GPU_DENSE 512
#else
    #define ROWS_IN_BLOCK 512
    #define GPU_DENSE 512
#endif

#define ITER 1
#define CPU_DENSE 8
#define SYNC_MS 2
#define __SYNC usleep(SYNC_MS)

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

struct storedBlock {
    int queue_Info;
    long int start_Loc; // Can Remove this Field
    long int end_Loc;
};
typedef struct storedBlock storedBlock;

struct threadArgs {
    int block_Index;
    bool working;
    bool is_Buf1;
    bool finished;
};
typedef struct threadArgs threadArgs;

struct writeArgs {
    int block_Index;
    bool working;
    bool is_Buf1;
    bool finished;
};
typedef struct writeArgs writeArgs;

extern float memcpy_Time1, memcpy_Time2, memcpy_Time3;

extern float elapsed_Time[7];

extern int mem_CPU, mem_GPU;

extern long int neC;
extern long int ne_Buffer;

extern csr A, B;
//extern std::string input_FileA, input_FileB;

extern int *d_PtrA, *d_IdxA, *d_PtrB, *d_IdxB;
extern float *d_ValA, *d_ValB;

extern int *h_Upp, *d_Upp;

extern int *block_Ptr;
extern int *block_Info;

extern int *ptrC;

//extern bool bufInfo;
extern int *cpu_BufIdx1, *gpu_BufIdx1;
extern float *cpu_BufVal1, *gpu_BufVal1;
extern int *cpu_BufIdx2, *gpu_BufIdx2;
extern float *cpu_BufVal2, *gpu_BufVal2;

extern int *h_IdxC;
extern float *h_ValC;
extern int *cpu_Ptr_for_Collecting;
extern int *d_PtrC, *d_IdxC;
extern float *d_ValC;
extern int *gpu_Ptr_for_Collecting;
extern int *d_TempIdx;
extern float *d_TempVal;

extern float *cpu_Dense;
extern float *gpu_Dense;

extern threadArgs *cpu_Args;
extern threadArgs *gpu_Args;
extern threadArgs *write_Args;

extern long int cpu_BufLoc;
extern long int gpu_BufLoc;

extern storedBlock *block_Queue;

extern long int ne_CPU;
extern long int ne_GPU;
extern int write_Count;

// For Device to Host Transfer Evaluation
extern int *gpu_BufIdx;
extern float *gpu_BufVal;

void read_CSR(std::string, csr *);

void initialize_GPU();
void initialize_A();
void initialize_B();
#ifndef COMPUTE
int initialize_C(int, int);
#else
int initialize_C(int);
#endif

void free_GPU();

void get_UpperBound();

void distribute_Workload();

void* compute_CPU(void *);
void* compute_GPU(void *);

void *manage_Buffer(void *);

void* write_File(void *);
void write_Ptr();
