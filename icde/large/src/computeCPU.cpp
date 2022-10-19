/**
 * Performs Sparse Matrix-Sparse Matrix Multiplication of one block in Matrix A
 * row_Start: Index of first row in this block
 * shmid_Flag: Flag which sends message to main thread that computation finished
 */

#include "SSpMM.hpp"


void* compute_CPU(void *args) {
    struct timeval st, ed;

	while (cpu_Args->finished == false) {
		while (cpu_Args->working == false) {
			if (cpu_Args->finished == true) {
				return NULL;
			}
			__SYNC;
		}

        gettimeofday(&st, NULL);
		int block_Index = cpu_Args->block_Index;
        bool is_Buf1 = cpu_Args->is_Buf1;
        long int location = cpu_BufLoc;

        int row_Start = block_Ptr[block_Index];
        int nr_in_this_block = block_Ptr[block_Index + 1] - row_Start;

        //int ompThread = CPU_DENSE;

        //printf("Start Block %d with CPUs\n", block_Index);
#pragma omp parallel for num_threads(CPU_DENSE) schedule(dynamic, 1)
        for (int rowA = row_Start; rowA < row_Start + nr_in_this_block; ++rowA) {
            int thread_Id = omp_get_thread_num();
            float *subRes = cpu_Dense + (A.nc * thread_Id);
            
            int num_NZs = 0;
            int elementC = h_Upp[rowA] - h_Upp[row_Start];
            int *ptr_IdxC = h_IdxC + elementC;
            float *ptr_ValC = h_ValC + elementC;
            int offset_Element = 0;

            for (int elementA = A.ptr[rowA]; elementA < A.ptr[rowA + 1]; ++elementA) {
                float valA = A.val[elementA];
                int rowB = A.idx[elementA];
                for (int elementB = B.ptr[rowB]; elementB < B.ptr[rowB + 1]; ++elementB) {
                    int col = B.idx[elementB];
                    if (subRes[col] < 0.0001 && subRes[col] > -0.0001) {
                        ptr_IdxC[num_NZs++] = col;
                    }
                    subRes[col] += valA * B.val[elementB];
                }
            }
//#ifndef COMPUTE
            for (int nz = 0; nz < num_NZs; ++nz) {
                int col = ptr_IdxC[nz];
                ptr_ValC[nz] = subRes[col];
                subRes[col] = 0.0f;
            }

            ptrC[rowA + 1] = num_NZs;
//#endif
        }

#ifndef COMPUTE
        int *addr_PtrC = ptrC + row_Start + 1;
        cpu_Ptr_for_Collecting[0] = 0;
#pragma unroll
        for (int row = 0; row < nr_in_this_block; ++row) {
            cpu_Ptr_for_Collecting[row + 1] = cpu_Ptr_for_Collecting[row] + addr_PtrC[row];
        }

        if (is_Buf1 == true) {
#pragma omp parallel for num_threads(CPU_DENSE) schedule(dynamic, 1)
            for (int row = 0; row < nr_in_this_block; ++row) {
                memcpy(cpu_BufIdx1 + location + cpu_Ptr_for_Collecting[row], h_IdxC + h_Upp[row_Start + row] - h_Upp[row_Start], sizeof(int) * addr_PtrC[row]);
                memcpy(cpu_BufVal1 + location + cpu_Ptr_for_Collecting[row], h_ValC + h_Upp[row_Start + row] - h_Upp[row_Start], sizeof(float) * addr_PtrC[row]);
            }
        }
        else {
#pragma omp parallel for num_threads(CPU_DENSE) schedule(dynamic, 1)
            for (int row = 0; row < nr_in_this_block; ++row) {
                memcpy(cpu_BufIdx2 + location + cpu_Ptr_for_Collecting[row], h_IdxC + h_Upp[row_Start + row] - h_Upp[row_Start], sizeof(int) * addr_PtrC[row]);
                memcpy(cpu_BufVal2 + location + cpu_Ptr_for_Collecting[row], h_ValC + h_Upp[row_Start + row] - h_Upp[row_Start], sizeof(float) * addr_PtrC[row]);
            }
        }
        
        block_Queue[block_Index] = {2, location, location + cpu_Ptr_for_Collecting[nr_in_this_block]};
        cpu_BufLoc += cpu_Ptr_for_Collecting[nr_in_this_block];
#endif

		gettimeofday(&ed, NULL);

		elapsed_Time[2] += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
		//printf("End Block %d with CPUs\n", block_Index);
        cpu_Args->working = false;
    }
}
