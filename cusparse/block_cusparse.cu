#include "cusparse.hpp"
#include <cusparse_v2.h>

void read_CSR(std::string , csr *);
void matMul(csr, csr, int);

int main (int argc, char **argv) {
    cudaFree(0);

    csr A;
    csr B;

    read_CSR(argv[1], &A);
    read_CSR(argv[1], &B);

    //matMul(A, B, atoi(argv[1]));
    matMul(A, B, 1);

}

void read_CSR(std::string input_File, csr *mat) {
    if (access((input_File + ".csr").c_str(), F_OK) < 0) {
        printf("No Such File\n");
        exit(0);
    }

    FILE *fp;
    fp = fopen((input_File + ".csr").c_str(), "rb");
    int row, col, edge;
    fread(&row, sizeof(int), 1, fp);
    fread(&col, sizeof(int), 1, fp);
    fread(&edge, sizeof(int), 1, fp);
    mat->nr = row;
    mat->nc = col;
    mat->ne = edge;

    mat->ptr = (int*)malloc(sizeof(int) * (mat->nr + 1));
    mat->idx = (int*)malloc(sizeof(int) * mat->ne);
    mat->val = (float*)malloc(sizeof(float) * mat->ne);

    fread(mat->ptr, sizeof(int), mat->nr + 1, fp);
    fread(mat->idx, sizeof(int), mat->ne, fp);
    fread(mat->val, sizeof(float), mat->ne, fp);
    fclose(fp);

    //printf("Input: %s\n\tNumber of Rows:\t\t%d\n\tNumber of Columns:\t%d\n\tNumber of Edges:\t%d\n\n", input_File.c_str(), mat->nr, mat->nc, mat->ne);
}

void filewrite(int *ptr, int *idx, float *val, int nr, int64_t ne) {
    FILE *fp = fopen("output", "wb");

    fwrite(ptr, sizeof(int), nr + 1, fp);
    fwrite(idx, sizeof(int), ne, fp);
    fwrite(val, sizeof(float), ne, fp);

    fflush(fp);
    fclose(fp);
}


void matMul(csr A, csr B, int mode) {
    float t_kernel = 0, t_d2h = 0, t_write = 0;
    struct timeval st, ed;
    long nnzC = 0;

    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t descrA = 0;
    cusparseSpMatDescr_t descrB = 0;
    cusparseSpMatDescr_t descrC = 0;

    cusparseSpGEMMDescr_t descr;

    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    float cons = 1.0;

    cusparseIndexType_t type1 = CUSPARSE_INDEX_32I;
    cusparseIndexType_t type2 = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;

    cudaDataType_t dType = CUDA_R_32F;

    int *ptrA, *idxA, *ptrB, *idxB;
    float *valA, *valB;

    cudaMalloc((void**)&ptrA, sizeof(int) * (A.nr +1));
    cudaMalloc((void**)&idxA, sizeof(int) * A.ne);
    cudaMalloc((void**)&valA, sizeof(float) * A.ne);
    cudaMalloc((void**)&ptrB, sizeof(int) * (B.nr +1));
    cudaMalloc((void**)&idxB, sizeof(int) * B.ne);
    cudaMalloc((void**)&valB, sizeof(float) * B.ne);

    cudaMemcpy(ptrB, B.ptr, sizeof(int) * (B.nr + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(idxA, A.idx, sizeof(int) * A.ne, cudaMemcpyHostToDevice);
    cudaMemcpy(valA, A.val, sizeof(float) * A.ne, cudaMemcpyHostToDevice);
    cudaMemcpy(idxB, B.idx, sizeof(int) * B.ne, cudaMemcpyHostToDevice);
    cudaMemcpy(valB, B.val, sizeof(float) * B.ne, cudaMemcpyHostToDevice);

    int *h_ptrC, *h_idxC;
    float *h_valC;
    
    int *ptrC, *idxC;
    float *valC;
    
    size_t bufSize1;
    size_t bufSize2;
    char *buf1, *buf2;
    int *temp_ptrA = (int*)malloc(sizeof(int) * (BLOCKSIZE + 1));

    printf("cuSPARSE\n");
    for (int i = 0; i < ITER; ++i) {
        nnzC = 0;
        // Single Block
    
        float t1 = 0, t2 = 0, t3 = 0;
        int num_Blocks  = ((A.nr - 1) / BLOCKSIZE) + 1;


        for (int i = 0; i < num_Blocks; ++i) {
            cusparseSpGEMM_createDescr(&descr);
            int nrA = BLOCKSIZE;
            int start_Row = i * BLOCKSIZE;
            int end_Row = (i + 1) * BLOCKSIZE;
            if (i == num_Blocks - 1) {
                nrA = A.nr - ((num_Blocks - 1) * BLOCKSIZE);
                end_Row = A.nr;
            }
            int *temp_idxA = idxA + A.ptr[i * BLOCKSIZE];
            float *temp_valA = valA + A.ptr[i * BLOCKSIZE];
            
            for (int j = 0; j < nrA + 1; ++j) {
                temp_ptrA[j] = A.ptr[start_Row + j] - A.ptr[start_Row];
            }
            cudaMemcpy(ptrA, temp_ptrA, sizeof(int) * (nrA + 1), cudaMemcpyHostToDevice);

            cusparseCreateCsr(&descrA, nrA, A.nc, temp_ptrA[nrA], ptrA, idxA + A.ptr[start_Row], valA + A.ptr[start_Row], type1, type2, base, dType);
            cusparseCreateCsr(&descrB, B.nr, B.nc, B.ne, ptrB, idxB, valB, type1, type2, base, dType);
            cusparseCreateCsr(&descrC, nrA, B.nc, 0, NULL, NULL, NULL, type1, type2, base, dType);
            
            gettimeofday(&st, NULL);

            cusparseSpGEMM_workEstimation(handle, op, op, &cons, descrA, descrB, &cons, descrC, dType, \
            CUSPARSE_SPGEMM_DEFAULT, descr, &bufSize1, NULL);
            cudaMalloc((void**)&buf1, sizeof(char) * bufSize1);

            cusparseSpGEMM_workEstimation(handle, op, op, &cons, descrA, descrB, &cons, descrC, dType, \
                CUSPARSE_SPGEMM_DEFAULT, descr, &bufSize1, buf1);

            cusparseSpGEMM_compute(handle, op, op, &cons, descrA, descrB, &cons, descrC, dType, \
            CUSPARSE_SPGEMM_DEFAULT, descr, &bufSize2, NULL);
            cudaMalloc((void**)&buf2, sizeof(char) * bufSize2);

            cusparseSpGEMM_compute(handle, op, op, &cons, descrA, descrB, &cons, descrC, dType, \
                CUSPARSE_SPGEMM_DEFAULT, descr, &bufSize2, buf2);
            gettimeofday(&ed, NULL);
            t1 += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);

            int64_t row, col, edge;
            cusparseSpMatGetSize(descrC, &row, &col, &edge);
            nnzC += edge;
            //printf("%d %d %d\n", row, col, edge);

            cudaMalloc((void**)&ptrC, sizeof(int) * (row + 1));
            cudaMalloc((void**)&idxC, sizeof(int) * edge);
            cudaMalloc((void**)&valC, sizeof(float) * edge);

            cusparseCsrSetPointers(descrC, ptrC, idxC, valC);

            cusparseSpGEMM_copy(handle, op, op, &cons, descrA, descrB, &cons, descrC, dType, \
            CUSPARSE_SPGEMM_DEFAULT, descr);
            cudaDeviceSynchronize();

            cudaFree(buf1);
            cudaFree(buf2);
            cusparseDestroySpMat(descrA);
            cusparseDestroySpMat(descrB);
            cusparseDestroySpMat(descrC);

            h_ptrC = (int*)malloc(sizeof(int) * (row + 1));
            h_idxC = (int*)malloc(sizeof(int) * edge);
            h_valC = (float*)malloc(sizeof(float) * edge);

            cudaMemcpy(h_ptrC, ptrC, sizeof(int) * (row + 1), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_idxC, idxC, sizeof(int) * edge, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_valC, valC, sizeof(float) * edge, cudaMemcpyDeviceToHost);

            cudaDeviceSynchronize();
            gettimeofday(&ed, NULL);
            t2 += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
            
            filewrite(h_ptrC, h_idxC, h_valC, row, edge);
            gettimeofday(&ed, NULL);
            t3 += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
            
            free(h_ptrC);
            free(h_idxC);
            free(h_valC);
            

            cudaFree(ptrC);
            cudaFree(idxC);
            cudaFree(valC);

            cusparseSpGEMM_destroyDescr(descr);
        }
        printf("%f\t%f\t%f\n", t1, t2, t3);
        t_kernel += t1;
        t_d2h += t2;
        t_write += t3;
        
    }
    printf("\nTotal nnzC %ld\n", nnzC);
    printf("Average:\n%f\t%f\t%f\n", t_kernel / ITER, t_d2h / ITER, t_write / ITER);
    cusparseDestroy(handle);

        free(temp_ptrA);
    free(A.ptr);
    free(A.idx);
    free(A.val);
    free(B.ptr);
    free(B.idx);
    free(B.val);

    cudaFree(ptrA);
    cudaFree(idxA);
    cudaFree(valA);
    cudaFree(ptrB);
    cudaFree(idxB);
    cudaFree(valB);


}
