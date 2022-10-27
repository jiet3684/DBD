#include "SSpMM.hpp"

int *d_PtrA, *d_IdxA, *d_PtrB, *d_IdxB;
float *d_ValA, *d_ValB;

int *d_Upp;

int *d_PtrC = NULL, *d_IdxC = NULL;
float *d_ValC = NULL;
int *d_TempIdx = NULL;
float *d_TempVal = NULL;
int *gpu_Ptr_for_Collecting = NULL;

float *gpu_Dense = NULL;

int *gpu_BufIdx;
float *gpu_BufVal;

//__global__ void coo2csr(edgeData *, int *, int *, float *, int);
__global__ void warm_up_gpu();
__global__ void calculate_UpperBound(int *, int *, int *, int *);

void initialize_GPU() {
    cudaFree(0);
    warm_up_gpu<<<4096, 512>>> ();
}

void initialize_A() {
    cudaMalloc((void**)&d_PtrA, sizeof(int) * (A.nr + 1));
    cudaMalloc((void**)&d_IdxA, sizeof(int) * A.ne);
    cudaMalloc((void**)&d_ValA, sizeof(float) * A.ne);

    cudaMemcpy(d_PtrA, A.ptr, sizeof(int) * (A.nr + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_IdxA, A.idx, sizeof(int) * A.ne, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ValA, A.val, sizeof(float) * A.ne, cudaMemcpyHostToDevice);
}

void initialize_B() {
    cudaMalloc((void**)&d_PtrB, sizeof(int) * (B.nr + 1));
    cudaMalloc((void**)&d_IdxB, sizeof(int) * B.ne);
    cudaMalloc((void**)&d_ValB, sizeof(float) * B.ne);

    cudaMemcpy(d_PtrB, B.ptr, sizeof(int) * (B.nr + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_IdxB, B.idx, sizeof(int) * B.ne, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ValB, B.val, sizeof(float) * B.ne, cudaMemcpyHostToDevice);
}

#ifndef COMPUTE
int initialize_C(int max_NeC, int multi_Blocks) {
    cudaMalloc((void**)&d_PtrC, sizeof(int) * (ROWS_IN_BLOCK + 1));
    cudaMalloc((void**)&d_IdxC, sizeof(int) * max_NeC);
    cudaMalloc((void**)&d_ValC, sizeof(float) * max_NeC);
    cudaMalloc((void**)&d_TempIdx, sizeof(int) * max_NeC);
    cudaMalloc((void**)&d_TempVal, sizeof(float) * max_NeC);

    cudaMalloc((void**)&gpu_Dense, sizeof(float) * GPU_DENSE * A.nc);
    if (gpu_Dense == NULL) {
        fprintf(stderr, "Cannot Allocate gpu_Dense Array.\n");
        return -1;
    }
    cudaMemset(gpu_Dense, 0, sizeof(float) * GPU_DENSE * A.nc);
    
    cudaMallocHost((void**)&ptrC, sizeof(int) * (A.nr + 1));

    cudaMallocHost((void**)&gpu_BufIdx1, sizeof(int) * (long)max_NeC * multi_Blocks);
    cudaMallocHost((void**)&gpu_BufVal1, sizeof(float) * (long)max_NeC * multi_Blocks);

    cudaMallocHost((void**)&gpu_BufIdx2, sizeof(int) * (long)max_NeC * multi_Blocks);
    cudaMallocHost((void**)&gpu_BufVal2, sizeof(float) * (long)max_NeC * multi_Blocks);

    cudaMallocHost((void**)&gpu_Ptr_for_Collecting, sizeof(int) * (ROWS_IN_BLOCK + 1));

    if ((ptrC == NULL) || (gpu_BufIdx1 == NULL) || (gpu_BufVal1 == NULL) || (gpu_BufIdx2 == NULL) || (gpu_BufVal2 == NULL) || (gpu_Ptr_for_Collecting == NULL)) {
        fprintf(stderr, "Cannot Allocate Buffers for Result from GPUs.\n");
        return -1;
    }
    return 0;
}
#else
int initialize_C(int max_NeC) {
    cudaMallocHost((void**)&ptrC, sizeof(int) * (A.nr + 1));
    h_IdxC = (int*)malloc(sizeof(int) * max_NeC);
    h_ValC = (float*)malloc(sizeof(float) * max_NeC);

    cpu_Dense = (float*)malloc(sizeof(float) * A.nc * CPU_DENSE);
    memset(cpu_Dense, 0, sizeof(float) * A.nc * CPU_DENSE);

    cudaMalloc((void**)&d_PtrC, sizeof(int) * (ROWS_IN_BLOCK + 1));
    cudaMalloc((void**)&d_IdxC, sizeof(int) * max_NeC);
    cudaMalloc((void**)&d_ValC, sizeof(float) * max_NeC);
    cudaMalloc((void**)&d_TempIdx, sizeof(int) * max_NeC);
    cudaMalloc((void**)&d_TempVal, sizeof(float) * max_NeC);

    cudaMallocHost((void**)&gpu_BufIdx, sizeof(int) * (A.nr + 1));
    cudaMallocHost((void**)&gpu_BufVal, sizeof(int) * max_NeC);
    cudaMallocHost((void**)&gpu_Ptr_for_Collecting, sizeof(int) * (ROWS_IN_BLOCK + 1));

    cudaMalloc((void**)&gpu_Dense, sizeof(float) * GPU_DENSE * A.nc);
    if (gpu_Dense == NULL) {
        fprintf(stderr, "Cannot Allocate gpu_Dense Array.\n");
        return -1;
    }
    cudaMemset(gpu_Dense, 0, sizeof(float) * GPU_DENSE * A.nc);
    
    return 0;
}
#endif

/*void convertCSR(const edgeData *edges, csr *mat, int *d_Ptr, int *d_Idx, float *d_Val) {
    mat->ptr = (int*)malloc(sizeof(int) * (mat->nr + 1));
    mat->idx = (int*)malloc(sizeof(int) * mat->ne);
    mat->val = (float*)malloc(sizeof(float) * mat->ne);
    
    edgeData *temp;
    cudaMalloc((void**)&temp, sizeof(edgeData) * mat->ne);
    cudaMemcpy(temp, edges, sizeof(edgeData) * mat->ne, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_Ptr, sizeof(int) * (mat->nr + 1));
    cudaMalloc((void**)&d_Idx, sizeof(int) * mat->ne);
    cudaMalloc((void**)&d_Val, sizeof(float) * mat->ne);

    cudaMemcpy(d_Ptr, mat->ptr, sizeof(int) * (mat->nr + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Idx, mat->idx, sizeof(int) * mat->ne, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Val, mat->val, sizeof(float) * mat->ne, cudaMemcpyHostToDevice);

    coo2csr<<<(mat->ne + 127) / 128, 128>>> (temp, d_Ptr, d_Idx, d_Val, mat->ne);

    cudaMemcpy(mat->ptr + 1, d_Ptr, sizeof(int) * mat->nr, cudaMemcpyDeviceToHost);
    cudaMemcpy(mat->idx, d_Idx, sizeof(int) * mat->ne, cudaMemcpyDeviceToHost);
    cudaMemcpy(mat->val, d_Val, sizeof(float) * mat->ne, cudaMemcpyDeviceToHost);

    mat->ptr[0] = 0;
    for (int row = 0; row < mat->nr; ++row) {
        mat->ptr[row + 1] += mat->ptr[row];
    }

    cudaMemcpy(d_Ptr, mat->ptr, sizeof(int) * (mat->nr + 1), cudaMemcpyDeviceToHost);

    cudaFree(temp);
}*/

void get_UpperBound() {
    h_Upp = (int*)malloc(sizeof(int) * (A.nr + 1));
    cudaMalloc((void**)&d_Upp, sizeof(int) * (A.nr + 1));
    cudaMemset(d_Upp, 0, sizeof(int) * (A.nr + 1));
    
    calculate_UpperBound<<<A.nr, 32>>> (d_PtrA, d_IdxA, d_PtrB, d_Upp);
    
    cudaMemcpy(h_Upp + 1, d_Upp, sizeof(int) * A.nr, cudaMemcpyDeviceToHost);
    h_Upp[0] = 0;
#pragma unroll
    for (int row = 0; row < A.nr; ++row) {
        h_Upp[row + 1] += h_Upp[row];
    }
    cudaMemcpy(d_Upp, h_Upp, sizeof(int) * A.nr, cudaMemcpyHostToDevice);
}

/*
__global__ void coo2csr(edgeData *edges, int *ptr, int *idx, float *val, int ne) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < ne) {
        idx[tid] = edges[tid].to;
        val[tid] = edges[tid].val;
        atomicAdd(ptr + edges[tid].from, 1);
    }
}
*/

__global__ void calculate_UpperBound(int *ptrA, int *idxA, int *ptrB, int *upp) {
	int thread_id = threadIdx.x;
	int baseA = ptrA[blockIdx.x];	// index of first element in A's row.
	int rowB;

	for (int i = baseA + thread_id; i < ptrA[blockIdx.x + 1]; i += blockDim.x)	{
		rowB = idxA[i];
		atomicAdd(&upp[blockIdx.x], ptrB[rowB + 1] - ptrB[rowB]); 
	}
}

__global__ void warm_up_gpu(){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid; 
  }