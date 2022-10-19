#include "cusparse.hpp"
#include <cusparse_v2.h>

void readInput(char *);
int sortEdges(const void *, const void *);
void convertCSR(edgeData *, edgeData *);

int *h_ptrC;
int *h_idxC;
float *h_valC;

cusparseHandle_t handle;


__global__ void calculate_UpperBound(int *ptrA, int *idxA, int *ptrB, int *upp) {
	int thread_id = threadIdx.x;
	int baseA = ptrA[blockIdx.x];	// index of first element in A's row.
	int rowB;

	for (int i = baseA + thread_id; i < ptrA[blockIdx.x + 1]; i += blockDim.x)	{
		rowB = idxA[i];
		atomicAdd(&upp[blockIdx.x], ptrB[rowB + 1] - ptrB[rowB]); 
	}
    //if (thread_id == 0) printf("%d\n", upp[blockIdx.x]);
}


__global__ void coo2csr(edgeData *, int *, int *, float *, int);

int *ptrA;
int *idxA;
float *valA;
int *ptrB;
int *idxB;
float *valB;

void readInput(char *input) {
    struct timeval st, ed;
    FILE *fp1, *fp2;
    fp1 = fopen(input, "rb");
    fp2 = fopen(input, "rb");

    char line[200];

    // Read Matrix A
    gettimeofday(&st, NULL);
    fgets(line, 200, fp1);
    bool symmetric = false;

    if(strstr(line, "symmetric")) symmetric = true;

        while (fgets(line, 200, fp1) != NULL) {
        if (line[0] == '%' || line[0] == '#') continue;   // Comment
        char delims[] = "\t, ";
        char *t;
        t = strtok(line, delims);
        if (t == NULL) {
            fprintf(stderr, "Wrong Format.\n");
            exit(-1);
        }
        A.nr = atoi(t) + 1;
        
        t = strtok(NULL, delims);
        if (t == NULL) {
            fprintf(stderr, "Wrong Format.\n");
            exit(-1);
        }
        A.nc = atoi(t) + 1;

        t = strtok(NULL, delims);
        A.ne = atoi(t);
        if (symmetric == true) A.ne *= 2;

        break;
    }

    //printf("Matrix A:\nNumber of Rows:\t\t%d\nNumber of Columns:\t%d\nNumber of Edges:\t%d\n\n", A.nr, A.nc, A.ne);

    edgeData *edgesA = (edgeData*)malloc(sizeof(edgeData) * A.ne);
    int offset = 0;
    while (fgets(line, 200, fp1) != NULL) {
        char delims[] = "\t, ";
        char *t;
        t = strtok(line, delims);
        edgesA[offset].from = atoi(t);

        t = strtok(NULL, delims);
        edgesA[offset].to = atoi(t);

        t = strtok(NULL, delims);
        if (t != NULL) edgesA[offset].val = atof(t);
        else edgesA[offset].val = 1.0f;

        if ((symmetric == true) && (edgesA[offset].from != edgesA[offset].to)) {
            edgesA[offset + 1].from = edgesA[offset].to;
            edgesA[offset + 1].to = edgesA[offset].from;
            edgesA[offset + 1].val = edgesA[offset].val;

            offset++;
        }
        offset++;
    }

    // Read Matrix B
    fgets(line, 200, fp2);
    symmetric = false;

    if(strstr(line, "symmetric")) symmetric = true;

        while (fgets(line, 200, fp2) != NULL) {
        if (line[0] == '%' || line[0] == '#') continue;   // Comment
        char delims[] = "\t, ";
        char *t;
        t = strtok(line, delims);
        if (t == NULL) {
            fprintf(stderr, "Wrong Format.\n");
            exit(-1);
        }
        B.nr = atoi(t) + 1;
        
        t = strtok(NULL, delims);
        if (t == NULL) {
            fprintf(stderr, "Wrong Format.\n");
            exit(-1);
        }
        B.nc = atoi(t) + 1;

        t = strtok(NULL, delims);
        B.ne = atoi(t);
        if (symmetric == true) B.ne *= 2;

        break;
    }

    //printf("Matrix B:\nNumber of Rows:\t\t%d\nNumber of Columns:\t%d\nNumber of Edges:\t%d\n\n", B.nr, B.nc, B.ne);

    edgeData *edgesB = (edgeData*)malloc(sizeof(edgeData) * B.ne);
    offset = 0;
    while (fgets(line, 200, fp2) != NULL) {
        char delims[] = "\t, ";
        char *t;
        t = strtok(line, delims);
        edgesB[offset].from = atoi(t);

        t = strtok(NULL, delims);
        edgesB[offset].to = atoi(t);

        t = strtok(NULL, delims);
        if (t != NULL) edgesB[offset].val = atof(t);
        else edgesB[offset].val = 1.0f;

        if ((symmetric == true) && (edgesB[offset].from != edgesB[offset].to)) {
            edgesB[offset + 1].from = edgesB[offset].to;
            edgesB[offset + 1].to = edgesB[offset].from;
            edgesB[offset + 1].val = edgesB[offset].val;

            offset++;
        }
        offset++;
    }
    gettimeofday(&ed, NULL);
    printf("Read Input File: %f\n", (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec));


    gettimeofday(&st, NULL);
    qsort((void*)edgesA, A.ne, sizeof(edgeData), sortEdges);
    qsort((void*)edgesB, B.ne, sizeof(edgeData), sortEdges);
    gettimeofday(&ed, NULL);
    printf("Sort Edges: %f\n", (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec));

    convertCSR(edgesA, edgesB);

    

    free(edgesA);
    free(edgesB);
    fclose(fp1);
    fclose(fp2);
}

int sortEdges(const void *a, const void *b) {
    edgeData dataA = *((edgeData*)a);
    edgeData dataB = *((edgeData*)b);

    if (dataA.from != dataB.from)
        return dataA.from - dataB.from;
    else if (dataA.to != dataB.to)
        return dataA.to - dataB.to;
    else if (dataA.val > dataB.val)
        return 1;
    return -1;
}

void convertCSR(edgeData *edgesA, edgeData *edgesB) {
    struct timeval st, ed;

    gettimeofday(&st, NULL);
    A.ptr = (int*)malloc(sizeof(int) * (A.nr + 1));
    A.idx = (int*)malloc(sizeof(int) * A.ne);
    A.val = (float*)malloc(sizeof(float) * A.ne);

    A.ptr[0] = 0;
    A.ptr[A.nr] = A.ne;
    int row_Offset = 0;
    int edges_in_row = 0;
    for (int e = 0; e < A.ne; ++e) {
        if (edgesA[e].from > row_Offset) {
            A.ptr[row_Offset] = edges_in_row;

            for (int empty_Rows = row_Offset + 1; empty_Rows < edgesA[e].from; ++empty_Rows) {
                A.ptr[empty_Rows + 1] = A.ptr[empty_Rows];
            }

            row_Offset = edgesA[e].from;
            edges_in_row = 0;
        }
        A.idx[e] = edgesA[e].to;
        A.val[e] = edgesA[e].val;
        edges_in_row++;
    }

    B.ptr = (int*)malloc(sizeof(int) * (B.nr + 1));
    B.idx = (int*)malloc(sizeof(int) * B.ne);
    B.val = (float*)malloc(sizeof(float) * B.ne);

    B.ptr[0] = 0;
    B.ptr[A.nr] = B.ne;
    row_Offset = 0;
    edges_in_row = 0;
    for (int e = 0; e < B.ne; ++e) {
        if (edgesB[e].from > row_Offset) {
            B.ptr[row_Offset] = edges_in_row;

            for (int empty_Rows = row_Offset + 1; empty_Rows < edgesB[e].from; ++empty_Rows) {
                B.ptr[empty_Rows + 1] = B.ptr[empty_Rows];
            }

            row_Offset = edgesB[e].from;
            edges_in_row = 0;
        }
        B.idx[e] = edgesB[e].to;
        B.val[e] = edgesB[e].val;
        edges_in_row++;
    }

    cudaMalloc((void**)&ptrA, sizeof(int) * (A.nr + 1));
    cudaMalloc((void**)&idxA, sizeof(int) * A.ne);
    cudaMalloc((void**)&valA, sizeof(float) * A.ne);
    cudaMalloc((void**)&ptrB, sizeof(int) * (B.nr + 1));
    cudaMalloc((void**)&idxB, sizeof(int) * B.ne);
    cudaMalloc((void**)&valB, sizeof(float) * B.ne);

    cudaMemcpy(ptrA, A.ptr, sizeof(int) * (A.nr + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(idxA, A.idx, sizeof(int) * A.ne, cudaMemcpyHostToDevice);
    cudaMemcpy(valA, A.val, sizeof(float) * A.ne, cudaMemcpyHostToDevice);
    cudaMemcpy(ptrB, B.ptr, sizeof(int) * (B.nr + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(idxB, B.idx, sizeof(int) * B.ne, cudaMemcpyHostToDevice);
    cudaMemcpy(valB, B.val, sizeof(float) * B.ne, cudaMemcpyHostToDevice);

    gettimeofday(&ed, NULL);
    printf("Convert to CSR Format(CPU): %f\n", (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec));
}

void filewrite(int *ptr, int *idx, float *val, int nr, int ne) {
    FILE *fp = fopen("output", "wb");

    fwrite(ptr, sizeof(int), nr + 1, fp);
    fwrite(idx, sizeof(int), ne, fp);
    fwrite(val, sizeof(float), ne, fp);

    fclose(fp);
}

void matMul(csr A, csr B) {
    float kernel_t = 0, d2h_t = 0, write_t = 0;
    struct timeval st, ed;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t descrA = 0;
    cusparseSpMatDescr_t descrB = 0;
    cusparseSpMatDescr_t descrC = 0;

    int num_Blocks = ((A.nr - 1) / 256) + 1;
    size_t max_Bufsize1 = 0, max_Bufsize2 = 0;


    int *ptrC, *idxC;
    float *valC;
    
    size_t bufSize1;
    size_t bufSize2;
    char *buf1, *buf2;

    cusparseSpGEMMDescr_t descr;
    cusparseSpGEMM_createDescr(&descr);
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    float cons = 1.0;

    cusparseIndexType_t type1 = CUSPARSE_INDEX_32I;
    cusparseIndexType_t type2 = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;

    cudaDataType_t dType = CUDA_R_32F;

    gettimeofday(&st, NULL);
    for (int i = 0; i < num_Blocks - 1; ++t) {
        cusparseCreateCsr(&descrA, 256, A.nc, A.ptr[i + 256] - A.ptr[i], ptrA + (i * 256), idxA + A.ptr[i * 256], valA + A.ptr[i * 256], type1, type2, base, dType);
        cusparseCreateCsr(&descrB, 256, B.nc, B.ptr[i + 256] - B.ptr[i], ptrB + (i * 256), idxB + B.ptr[i * 256], valB + B.ptr[i * 256], type1, type2, base, dType);
        cusparseCreateCsr(&descrC, 256, B.nc, 0, NULL, NULL, NULL, type1, type2, base, dType);

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

        int64_t row, col, edge;
        cusparseSpMatGetSize(descrC, &row, &col, &edge);
        //printf("%d %d %d\n", row, col, edge);

        cudaMalloc((void**)&ptrC, sizeof(int) * (row + 1));
        cudaMalloc((void**)&idxC, sizeof(int) * edge);
        cudaMalloc((void**)&valC, sizeof(float) * edge);

        cusparseCsrSetPointers(descrC, ptrC, idxC, valC);

        cusparseSpGEMM_copy(handle, op, op, &cons, descrA, descrB, &cons, descrC, dType, \
        CUSPARSE_SPGEMM_DEFAULT, descr);
    }
    // Final Block
    cudaDeviceSynchronize();

    gettimeofday(&ed, NULL);
    kernel_t += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);


#ifdef D2H
    h_ptrC = (int*)malloc(sizeof(int) * (row + 1));
    h_idxC = (int*)malloc(sizeof(int) * edge);
    h_valC = (float*)malloc(sizeof(float) * edge);

    gettimeofday(&st, NULL);

    cudaMemcpy(h_ptrC, ptrC, sizeof(int) * (row + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idxC, idxC, sizeof(int) * edge, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_valC, valC, sizeof(float) * edge, cudaMemcpyDeviceToHost);


    gettimeofday(&ed, NULL);
    d2h_t += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
    printf("Output NNZ %d Elements\n", edge);
#endif

#ifdef WRITE
    gettimeofday(&st, NULL);
    filewrite(h_ptrC, h_idxC, h_valC, A.nr, edge);
    gettimeofday(&ed, NULL);
    write_t += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
#endif

    printf("cuSPARSE\nCompute: %f\n", kernel_t);
#ifdef D2H
    printf("Compute + D2H: %f\n", kernel_t + d2h_t);
#endif
#ifdef WRITE
    printf("Compute + D2H + Write: %f\n", kernel_t + d2h_t + write_t);
#endif

#ifdef D2H
    free(h_ptrC);
    free(h_idxC);
    free(h_valC);

#endif



    cusparseSpGEMM_destroyDescr(descr);
    cusparseDestroySpMat(descrA);
    cusparseDestroySpMat(descrB);
    cusparseDestroySpMat(descrC);
    cusparseDestroy(handle);
    cudaFree(buf1);
    cudaFree(buf2);
    cudaFree(ptrC);
    cudaFree(idxC);
    cudaFree(valC);
}

void freeMem() {
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