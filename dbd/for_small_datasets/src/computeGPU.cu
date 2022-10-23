#include "SSpMM.hpp"

__global__ void rowproduct_ddense(int *csr_ptr_d1, int *csr_idx_d1, float *csr_val_d1, int *csr_ptr_d2, int *csr_idx_d2, float *csr_val_d2, int *upp, int *csr_idx_d3, float *csr_val_d3, int start);
__global__ void rowproduct_dense(int *csr_ptr_d1, int *csr_idx_d1, float *csr_val_d1, int *csr_ptr_d2, int *csr_idx_d2, float *csr_val_d2, int *upp, int *csr_idx_d3, float *csr_val_d3, int start);
__global__ void rowproduct_sparse(int *csr_ptr_d1, int *csr_idx_d1, float *csr_val_d1, int *csr_ptr_d2, int *csr_idx_d2, float *csr_val_d2, int *upp, int *csr_idx_d3, float *csr_val_d3, int start);
__global__ void merge(int* upp, int* ptr, int* idx, int *idx_output, float* val, float* dense, int start, int nr, int nc);
__global__ void collect(int* upp, int* ptr, int* idx_before, int* idx_after, float* val_before, float* val_after, int start);

inline int analyze_Block(int block_Index) {
    if (block_Info[block_Index] <= 128) return 64;
    else if (block_Info[block_Index] <= 512) return 256;
    return 1024;
}

void* compute_GPU(void *args) {
	struct timeval st, ed;

	while (gpu_Args->finished == false) {
		while (gpu_Args->working == false) {
			if (gpu_Args->finished == true) {
				return NULL;
			}
			__SYNC;
		}

		gettimeofday(&st, NULL);
		int block_Index = gpu_Args->block_Index;
		bool is_Buf1 = gpu_Args->is_Buf1;
		long int location = gpu_BufLoc;
    	//printf("\tStart Block %d with GPUs\n", block_Index);

		int row_Offset = block_Ptr[block_Index];
		int nr_in_this_block = block_Ptr[block_Index + 1] - row_Offset;

		int thread_Num = analyze_Block(block_Index);
		if (thread_Num == 64)           rowproduct_sparse<<<nr_in_this_block, thread_Num>>> (d_PtrA, d_IdxA, d_ValA, d_PtrB, d_IdxB, d_ValB, d_Upp, d_IdxC, d_TempVal, row_Offset);
		else if (thread_Num == 256)     rowproduct_dense<<<nr_in_this_block, thread_Num>>> (d_PtrA, d_IdxA, d_ValA, d_PtrB, d_IdxB, d_ValB, d_Upp, d_IdxC, d_TempVal, row_Offset);
		else if (thread_Num == 1024)    rowproduct_ddense<<<nr_in_this_block, thread_Num>>> (d_PtrA, d_IdxA, d_ValA, d_PtrB, d_IdxB, d_ValB, d_Upp, d_IdxC, d_TempVal, row_Offset);

		merge<<<GPU_DENSE, thread_Num>>> (d_Upp, d_PtrC, d_IdxC, d_TempIdx, d_TempVal, gpu_Dense, row_Offset, nr_in_this_block, A.nc);

#ifndef COMPUTE
		int *addr_PtrC = ptrC + row_Offset + 1;
		cudaMemcpy(addr_PtrC, d_PtrC, sizeof(int) * nr_in_this_block, cudaMemcpyDeviceToHost);
		gpu_Ptr_for_Collecting[0] = 0;
#pragma unroll
		for (int row = 0; row < nr_in_this_block; ++row) {
			gpu_Ptr_for_Collecting[row + 1] = gpu_Ptr_for_Collecting[row] + addr_PtrC[row];
		}
		cudaMemcpy(d_PtrC, gpu_Ptr_for_Collecting, sizeof(int) * (nr_in_this_block + 1), cudaMemcpyHostToDevice);
	
		collect<<<nr_in_this_block, thread_Num>>> (d_Upp, d_PtrC, d_TempIdx, d_IdxC, d_TempVal, d_ValC, row_Offset);

		if (is_Buf1 == true) {
			cudaMemcpy(gpu_BufIdx1 + location, d_IdxC, sizeof(int) * gpu_Ptr_for_Collecting[nr_in_this_block], cudaMemcpyDeviceToHost);
			cudaMemcpy(gpu_BufVal1 + location, d_ValC, sizeof(float) * gpu_Ptr_for_Collecting[nr_in_this_block], cudaMemcpyDeviceToHost);
		}
		else {
			cudaMemcpy(gpu_BufIdx2 + location, d_IdxC, sizeof(int) * gpu_Ptr_for_Collecting[nr_in_this_block], cudaMemcpyDeviceToHost);
			cudaMemcpy(gpu_BufVal2 + location, d_ValC, sizeof(float) * gpu_Ptr_for_Collecting[nr_in_this_block], cudaMemcpyDeviceToHost);
		}
		
		block_Queue[block_Index] = {1, location, location + gpu_Ptr_for_Collecting[nr_in_this_block]};
		gpu_BufLoc += gpu_Ptr_for_Collecting[nr_in_this_block];
#endif

#ifdef D2H
		int *addr_PtrC = ptrC + row_Offset + 1;
		cudaMemcpy(addr_PtrC, d_PtrC, sizeof(int) * nr_in_this_block, cudaMemcpyDeviceToHost);
		gpu_Ptr_for_Collecting[0] = 0;
#pragma unroll
		for (int row = 0; row < nr_in_this_block; ++row) {
			gpu_Ptr_for_Collecting[row + 1] = gpu_Ptr_for_Collecting[row] + addr_PtrC[row];
		}
		cudaMemcpy(d_PtrC, gpu_Ptr_for_Collecting, sizeof(int) * (nr_in_this_block + 1), cudaMemcpyHostToDevice);

		collect<<<nr_in_this_block, thread_Num>>> (d_Upp, d_PtrC, d_TempIdx, d_IdxC, d_TempVal, d_ValC, row_Offset);

		cudaMemcpy(gpu_BufIdx, d_IdxC, sizeof(int) * gpu_Ptr_for_Collecting[nr_in_this_block], cudaMemcpyDeviceToHost);
		cudaMemcpy(gpu_BufVal, d_ValC, sizeof(float) * gpu_Ptr_for_Collecting[nr_in_this_block], cudaMemcpyDeviceToHost);

		/*int *addr_PtrC = ptrC + row_Offset + 1;
		cudaMemcpy(addr_PtrC, d_PtrC, sizeof(int) * nr_in_this_block, cudaMemcpyDeviceToHost);
		
		cudaMemcpy(gpu_BufIdx, d_IdxC, sizeof(int) * (h_Upp[row_Offset + nr_in_this_block] - h_Upp[row_Offset]), cudaMemcpyDeviceToHost);
		cudaMemcpy(gpu_BufVal, d_ValC, sizeof(float) * (h_Upp[row_Offset + nr_in_this_block] - h_Upp[row_Offset]), cudaMemcpyDeviceToHost);*/

#endif

		gettimeofday(&ed, NULL);
		

		elapsed_Time[3] += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
		//printf("\tEnd Block %d with GPUs\n", block_Index);
		gpu_Args->working = false;
	}
	
}


void free_GPU() {
    if (d_PtrA) cudaFree(d_PtrA);
    if (d_IdxA) cudaFree(d_IdxA);
    if (d_ValA) cudaFree(d_ValA);

	if (d_PtrB) cudaFree(d_PtrB);
    if (d_IdxB) cudaFree(d_IdxB);
    if (d_ValB) cudaFree(d_ValB);

	if (d_Upp) cudaFree(d_Upp);
    if (d_TempIdx) cudaFree(d_TempIdx);
    if (d_TempVal) cudaFree(d_TempVal);

    if (d_PtrC) cudaFree(d_PtrC);
    if (d_IdxC) cudaFree(d_IdxC);
    if (d_ValC) cudaFree(d_ValC);

    if (gpu_Dense) cudaFree(gpu_Dense);

    if (ptrC) cudaFreeHost(ptrC);
    if (gpu_Ptr_for_Collecting) cudaFree(gpu_Ptr_for_Collecting);

#ifndef COMPUTE
    if (gpu_BufIdx1) cudaFreeHost(gpu_BufIdx1);
    if (gpu_BufVal1) cudaFreeHost(gpu_BufVal1);
    if (gpu_BufIdx2) cudaFreeHost(gpu_BufIdx2);
    if (gpu_BufVal2) cudaFreeHost(gpu_BufVal2);
#else
	if (gpu_BufIdx) cudaFreeHost(gpu_BufIdx);
	if (gpu_BufVal) cudaFreeHost(gpu_BufVal);
#endif
}


__global__ void rowproduct_ddense(int *csr_ptr_d1, int *csr_idx_d1, float *csr_val_d1, int *csr_ptr_d2, int *csr_idx_d2, float *csr_val_d2, int *upp, int *csr_idx_d3, float *csr_val_d3, int start){
	int now_Row = start + blockIdx.x;
    int baseA = csr_ptr_d1[now_Row];
	int baseB;
	int baseC = upp[now_Row] - upp[start];
	int valA;
	int len;
	int rowB;
	int offset;

	__shared__ int shared_base;
	if (threadIdx.x == 0) shared_base = 0;
	__syncthreads();
	
	for (int i = baseA + threadIdx.x; i < csr_ptr_d1[now_Row + 1]; i += blockDim.x) {
		valA = csr_val_d1[i];
		rowB = csr_idx_d1[i];
		baseB = csr_ptr_d2[rowB];
		int len = csr_ptr_d2[rowB + 1] - baseB;
		offset = atomicAdd(&shared_base, len);

		int len_div8 = (len >> 3) << 3;
		int len_div4 = (len >> 2) << 2;
		int len_div2 = (len >> 1) << 1;
		int idx1 = baseC + offset;
		int idx2 = baseB;
		while (idx1 < baseC + offset + len_div8) {
			csr_idx_d3[idx1] = csr_idx_d2[idx2];
			csr_idx_d3[idx1 + 1] = csr_idx_d2[idx2 + 1];
			csr_idx_d3[idx1 + 2] = csr_idx_d2[idx2 + 2];
			csr_idx_d3[idx1 + 3] = csr_idx_d2[idx2 + 3];
			csr_idx_d3[idx1 + 4] = csr_idx_d2[idx2 + 4];
			csr_idx_d3[idx1 + 5] = csr_idx_d2[idx2 + 5];
			csr_idx_d3[idx1 + 6] = csr_idx_d2[idx2 + 6];
			csr_idx_d3[idx1 + 7] = csr_idx_d2[idx2 + 7];

			csr_val_d3[idx1] = csr_val_d2[idx2] * valA;
			csr_val_d3[idx1 + 1] = csr_val_d2[idx2 + 1] * valA;
			csr_val_d3[idx1 + 2] = csr_val_d2[idx2 + 2] * valA;
			csr_val_d3[idx1 + 3] = csr_val_d2[idx2 + 3] * valA;
			csr_val_d3[idx1 + 4] = csr_val_d2[idx2 + 4] * valA;
			csr_val_d3[idx1 + 5] = csr_val_d2[idx2 + 5] * valA;
			csr_val_d3[idx1 + 6] = csr_val_d2[idx2 + 6] * valA;
			csr_val_d3[idx1 + 7] = csr_val_d2[idx2 + 7] * valA;

			idx1 += 8;
			idx2 += 8;
		}
		if (len_div8 < len_div4) {
			csr_idx_d3[idx1] = csr_idx_d2[idx2];
			csr_idx_d3[idx1 + 1] = csr_idx_d2[idx2 + 1];
			csr_idx_d3[idx1 + 2] = csr_idx_d2[idx2 + 2];
			csr_idx_d3[idx1 + 3] = csr_idx_d2[idx2 + 3];

			csr_val_d3[idx1] = csr_val_d2[idx2] * valA;
			csr_val_d3[idx1 + 1] = csr_val_d2[idx2 + 1] * valA;
			csr_val_d3[idx1 + 2] = csr_val_d2[idx2 + 2] * valA;
			csr_val_d3[idx1 + 3] = csr_val_d2[idx2 + 3] * valA;

			idx1 += 4;
			idx2 += 4;
		}
		if (len_div4 < len_div2) {
			csr_idx_d3[idx1] = csr_idx_d2[idx2];
			csr_idx_d3[idx1 + 1] = csr_idx_d2[idx2 + 1];

			csr_val_d3[idx1] = csr_val_d2[idx2] * valA;
			csr_val_d3[idx1 + 1] = csr_val_d2[idx2 + 1] * valA;
			idx1 += 2;	idx2 += 2;
		}
		if (len_div2 < len) {
			csr_idx_d3[idx1] = csr_idx_d2[idx2];
			csr_val_d3[idx1] = csr_val_d2[idx2] * valA;
		}
		
	}
}

__global__ void rowproduct_dense(int *csr_ptr_d1, int *csr_idx_d1, float *csr_val_d1, int *csr_ptr_d2, int *csr_idx_d2, float *csr_val_d2, int *upp, int *csr_idx_d3, float *csr_val_d3, int start){
	int now_Row = start + blockIdx.x;
    int baseA = csr_ptr_d1[now_Row];
	int baseB;
	int baseC = upp[now_Row] - upp[start];
	int valA;
	int len;
	int rowB;
	int offset;

	__shared__ int shared_base;
	if (threadIdx.x == 0) shared_base = 0;
	__syncthreads();
	
	for (int i = baseA + threadIdx.x; i < csr_ptr_d1[now_Row + 1]; i += blockDim.x) {
		valA = csr_val_d1[i];
		rowB = csr_idx_d1[i];
		baseB = csr_ptr_d2[rowB];
		int len = csr_ptr_d2[rowB + 1] - baseB;
		offset = atomicAdd(&shared_base, len);

		int len_div4 = (len >> 2) << 2;
		int len_div2 = (len >> 1) << 1;
		int idx1 = baseC + offset;
		int idx2 = baseB;
		while (idx1 < baseC + offset + len_div4) {
			csr_idx_d3[idx1] = csr_idx_d2[idx2];
			csr_idx_d3[idx1 + 1] = csr_idx_d2[idx2 + 1];
			csr_idx_d3[idx1 + 2] = csr_idx_d2[idx2 + 2];
			csr_idx_d3[idx1 + 3] = csr_idx_d2[idx2 + 3];

			csr_val_d3[idx1] = csr_val_d2[idx2] * valA;
			csr_val_d3[idx1 + 1] = csr_val_d2[idx2 + 1] * valA;
			csr_val_d3[idx1 + 2] = csr_val_d2[idx2 + 2] * valA;
			csr_val_d3[idx1 + 3] = csr_val_d2[idx2 + 3] * valA;

			idx1 += 4;
			idx2 += 4;
		}
		if (len_div4 < len_div2) {
			csr_idx_d3[idx1] = csr_idx_d2[idx2];
			csr_idx_d3[idx1 + 1] = csr_idx_d2[idx2 + 1];

			csr_val_d3[idx1] = csr_val_d2[idx2] * valA;
			csr_val_d3[idx1 + 1] = csr_val_d2[idx2 + 1] * valA;
			idx1 += 2;	idx2 += 2;
		}
		if (len_div2 < len) {
			csr_idx_d3[idx1] = csr_idx_d2[idx2];
			csr_val_d3[idx1] = csr_val_d2[idx2] * valA;
		}
		
	}
}

__global__ void rowproduct_sparse(int *csr_ptr_d1, int *csr_idx_d1, float *csr_val_d1, int *csr_ptr_d2, int *csr_idx_d2, float *csr_val_d2, int *upp, int *csr_idx_d3, float *csr_val_d3, int start) {
	int now_Row = start + blockIdx.x;
    int baseA = csr_ptr_d1[now_Row];
	int baseB;
	int baseC = upp[now_Row] - upp[start];
	int valA;
	int len;
	int rowB;
	int offset;

	__shared__ int shared_base;
	if (threadIdx.x == 0) shared_base = 0;
	__syncthreads();
	
	for (int i = baseA + threadIdx.x; i < csr_ptr_d1[now_Row + 1]; i += blockDim.x) {
		valA = csr_val_d1[i];
		rowB = csr_idx_d1[i];
		baseB = csr_ptr_d2[rowB];
		int len = csr_ptr_d2[rowB + 1] - baseB;
		offset = atomicAdd(&shared_base, len);
		
		for (int j = 0; j < len; ++j ) {
			csr_idx_d3[baseC + offset + j] = csr_idx_d2[baseB + j];
			csr_val_d3[baseC + offset + j] = csr_val_d2[baseB + j] * valA;
		}
	}
}

__global__ void merge(int* upp, int* ptr, int* idx, int *idx_output, float* val, float* dense, int start, int nr, int nc) {
    __shared__ int nnz;
	int baseDense = blockIdx.x * nc;

    for (int r = 0; r <= (nr / gridDim.x); ++r) {
		int bid = blockIdx.x + r * gridDim.x;
		if (bid >= nr) return;

		int baseC = upp[start + bid] - upp[start];
		
		if (threadIdx.x == 0) nnz = 0;
		__syncthreads();

		for (int i = baseC + threadIdx.x; i < upp[start + bid + 1] - upp[start]; i += blockDim.x) {
			int col = idx[i];
			float zero = atomicAdd(&dense[baseDense + col], val[i]);
			if (zero < 0.0001 && zero > -0.0001) {
				int loc = atomicAdd(&nnz, 1);
				idx_output[baseC + loc] = col;
			}
		}
		__syncthreads();
		
		for (int i = threadIdx.x; i < nnz; i += blockDim.x) {
			int col = idx_output[baseC + i];
			
			val[baseC + i] = dense[baseDense + col];
			dense[baseDense + col] = 0;
		}

		__syncthreads();
		if (threadIdx.x == 0) ptr[bid] = nnz;
	}
}

__global__ void collect(int* upp, int* ptr, int* idx_before, int* idx_after, float* val_before, float* val_after, int start) {
	int tid = threadIdx.x;
	int base_before = upp[start + blockIdx.x] - upp[start];
	int base_after = ptr[blockIdx.x];
	int len = ptr[blockIdx.x + 1] - base_after;

	for (int i = tid; i < len; i += blockDim.x) {
		idx_after[base_after + i] = idx_before[base_before + i];
		val_after[base_after + i] = val_before[base_before + i];
	}
}
