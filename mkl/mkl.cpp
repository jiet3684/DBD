#include "mkl.hpp"


void read_CSR(std::string , CSR *);
void matMul(int);

CSR A, B;


int main(int argc, char **argv) {
    struct timeval st, ed;
    gettimeofday(&st, NULL);

    read_CSR(argv[1], &A);
    read_CSR(argv[1], &B);

    matMul(0);

   
}

void read_CSR(std::string input_File, CSR *mat) {
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

void filewrite(int *ptr, int *idx, float *val, int nr) {
    FILE *fp = fopen("output", "wb");
    
    fwrite(ptr, sizeof(int), nr, fp);
    fwrite(idx, sizeof(int), ptr[nr - 1], fp);
    fwrite(val, sizeof(float), ptr[nr - 1], fp);
    
    fclose(fp);
}

void matMul(int mode) {
    float t_kernel = 0, t_write = 0;
    struct timeval st, ed;

    sparse_matrix_t sp_A, sp_B, sp_C = NULL;
    sparse_status_t status;

    long nnzC = 0;
    printf("MKL\n");
    for (int i = 0; i < ITER; ++i) {
        nnzC = 0;
        if (mode == 0) {
            gettimeofday(&st, NULL);
            status = mkl_sparse_s_create_csr(&sp_A, SPARSE_INDEX_BASE_ZERO, A.nr, A.nc, A.ptr, A.ptr + 1, A.idx, A.val);
            status = mkl_sparse_s_create_csr(&sp_B, SPARSE_INDEX_BASE_ZERO, B.nr, B.nc, B.ptr, B.ptr + 1, B.idx, B.val);

            
            //mkl_sparse_optimize(sp_A);

            status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, sp_A, sp_B, &sp_C);

            mkl_sparse_order(sp_C);

            sparse_matrix_t output;
            sparse_index_base_t base;

            MKL_INT nr, nc;
            MKL_INT *row_start, *row_end;
            MKL_INT *idx;
            float *val;

            mkl_sparse_convert_csr(sp_C, SPARSE_OPERATION_NON_TRANSPOSE, &output);
            mkl_sparse_s_export_csr(output, &base, &nr, &nc, &row_start, &row_end, &idx, &val);
            gettimeofday(&ed, NULL);
            t_kernel += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
            printf("%f\t", (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec));

#ifdef WRITE
            filewrite(row_start, idx, val, nr);
            gettimeofday(&ed, NULL);
            t_write += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
            printf("%f\n", (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec));
#endif
            //printf("nr %d nc %d ne %d\n", nr, nc, row_end[nr - 1]);
            nnzC += row_end[nr - 1];

            mkl_sparse_destroy(sp_A);
            mkl_sparse_destroy(sp_B);
            mkl_sparse_destroy(sp_C);
        }
        else {
            int num_Blocks  = ((A.nr - 1) / BLOCKSIZE) + 1;

            int temp_ptrA[257];

            for (int i = 0; i < num_Blocks; ++i) {
                int nrA = BLOCKSIZE;
                int start_Row = i * BLOCKSIZE;
                int end_Row = (i + 1) * BLOCKSIZE;
                if (i == num_Blocks - 1) {
                    nrA = A.nr - ((num_Blocks - 1) * BLOCKSIZE);
                    end_Row = A.nr;
                }
            }


        }
    }

    printf("\nTotal nnzC %ld\n", nnzC);
    printf("Average:\n%f\t%f\n", t_kernel / ITER, t_write / ITER);

    //if (row_start) delete row_start;
    //if (row_end) free(row_end);
    //if (idx) free(idx);
    //if (val) free(val);
}


void freeMem() {
    free(A.ptr);
    free(A.idx);
    free(A.val);
    free(B.ptr);
    free(B.idx);
    free(B.val);

}