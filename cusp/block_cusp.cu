#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/print.h>
#include <cusp/functional.h>
#include <sys/time.h>
#include <vector>

#include <iostream>

#include "custom/timer.h"

__global__ void calc_mem(int *a_cidx , int* a_ptr,
                         int *b_cidx , int* b_ptr,
                         int *c_ptr){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int bsize = blockDim.x;

    int a_base = a_ptr[bid];
    int a_len = a_ptr[bid+1] - a_ptr[bid];
    for(int i = tid; i < a_len; i+=bsize){
        int col = a_cidx[a_base + i];
        int b_len = b_ptr[col+1] - b_ptr[col];
        atomicAdd(&c_ptr[bid], b_len);
    }
}

int *a_ptr;
int *a_ridx;
int *a_cidx;
int dimension;
int nnz;

struct cooData
{
	int ridx;
	int cidx;
	float val;
};
typedef struct cooData cooData;

struct coo
{
	cooData* data;

	int n;
	int e;
};
typedef struct coo coo;

struct csr
{
	int *header;
	int *cidx;
	float *val;

	int n;
	int e;
};
typedef struct csr csr;

void freeCOO( coo* mat ){
	if( mat->data != NULL )
		free( mat->data );

	mat->data = NULL;
	mat->e = 0;
	mat->n = 0;
}

coo copyCOO( coo mat ){
	coo ret;

	if( mat.e != 0 ){
		ret.data = (cooData*)malloc( sizeof(cooData) * mat.e );
		memcpy( ret.data, mat.data, sizeof(cooData) * mat.e );
	}
	else{
		ret.data = NULL;
	}

	ret.e = mat.e;
	ret.n = mat.n;

	return ret;
}

int cooCompareByRow( const void* a, const void* b){
	cooData dataA = *(cooData*)a;
	cooData dataB = *(cooData*)b;

	if( dataA.ridx != dataB.ridx )
		return dataA.ridx - dataB.ridx;

	else if( dataA.cidx != dataB.cidx )
		return dataA.cidx - dataB.cidx;

	else if( dataA.val > dataB.val )
		return 1;
	else
		return -1;
}

int cooCompareByCol( const void* a, const void* b)
{
	cooData dataA = *(cooData*)a;
	cooData dataB = *(cooData*)b;

	if( dataA.cidx != dataB.cidx )
		return dataA.cidx - dataB.cidx;

	else if( dataA.ridx != dataB.ridx )
		return dataA.ridx - dataB.ridx;
	else if( dataA.val > dataB.val )
		return 1;
	else
		return -1;
}


#define SORT_COO_ROW_BASE  1
#define SORT_COO_COL_BASE  0

int sortCOO( coo mat, int base ){
    if( base != SORT_COO_ROW_BASE && base != SORT_COO_COL_BASE )
		return -1;

	if( base == SORT_COO_ROW_BASE )
		qsort( (void*)mat.data, (size_t)mat.e, sizeof(cooData), cooCompareByRow );
	if( base == SORT_COO_COL_BASE )
		qsort( (void*)mat.data, (size_t)mat.e, sizeof(cooData), cooCompareByCol );


	return 0;
}


void readMTX( coo* mat, char*path ){
	FILE* fp = fopen(path,"r");
	int dim = 0;

    char line[100];
    int sym = 0;
    fgets(line,100,fp);
    if(strstr(line,"symmetric")) sym = 1;

    while(1){
        fgets(line,100,fp);
        if(line[0]!='%'){
            break;
        }
    }
	int n = atoi(strtok(line," \n\t"));
    int m = atoi(strtok(NULL," \n\t"));
    int e = atoi(strtok(NULL," \n\t"));
    
	if(sym) mat->data = (cooData*)malloc(sizeof(cooData)*e*2);
    else  mat->data = (cooData*)malloc(sizeof(cooData)*e);

    mat->e = 0;
    mat->n = n;
    int i=0;
    int r,c;
    float val;
    char fuck[200];

    while(fscanf(fp,"%d %d %s",&r,&c,fuck)!=EOF){

        if(sym){
            if(r==c){
                mat->data[i].ridx = r-1;
                mat->data[i].cidx = c-1;
                mat->data[i].val = 1.1;
                mat->e++; i++;
            }
            else{
                mat->data[i].ridx = r-1;
                mat->data[i].cidx = c-1;
                mat->data[i].val = 1.1;
                mat->data[i+1].ridx = c-1;
                mat->data[i+1].cidx = r-1;
                mat->data[i+1].val = 1.1;

                mat->e+=2; i+=2;
            }
        }
        else{
            mat->data[i].ridx = r-1;
            mat->data[i].cidx = c-1;
            mat->data[i].val = 1.1;
            mat->e++; i++;
        }
    }
    fclose(fp);
}

csr transformCOOToCSR( coo mat )
{
	csr ret;

	coo sorted = copyCOO( mat );
	//sortCOO( sorted, SORT_COO_ROW_BASE );
	if( sorted.e != 0 )
	{
		ret.cidx = (int*)malloc(sizeof(int) * sorted.e );
		ret.val = (float*)malloc(sizeof(float) * sorted.e );
		ret.header = (int*)malloc(sizeof(int) * (sorted.n+1) );
		int last_ridx = sorted.data[0].ridx;
		for( int i=0; i<=last_ridx; ++i )
			ret.header[i] = 0;

		for( int i=0; i<sorted.e; ++i )
		{
			if( last_ridx != sorted.data[i].ridx )
			{
				for( int j=last_ridx+1; j<=sorted.data[i].ridx; ++j )
				{
					ret.header[j] = i;
				}

				last_ridx = sorted.data[i].ridx;
			}
            
			ret.cidx[i] = sorted.data[i].cidx;
			ret.val[i] = sorted.data[i].val;
		}
		for( int j=last_ridx+1; j<sorted.n; ++j )
		{
			ret.header[j] = sorted.e; 
		}
	}
	else
	{
		ret.header = NULL;
		ret.cidx = NULL;
		ret.val = NULL;
	}
	ret.e = sorted.e;
	ret.n = sorted.n;
    ret.header[ret.n] = ret.e;

	freeCOO( &sorted );
	return ret;
}


void filewrite(int *ptr, int *idx, float *val, int nr, int ne) {
    FILE *fp = fopen("output", "wb");

    fwrite(ptr, sizeof(int), nr + 1, fp);
    fwrite(idx, sizeof(int), ne, fp);
    fwrite(val, sizeof(float), ne, fp);

    fclose(fp);
}

int main(int argc, char* argv[])
{
    // initialize matrix
    //cusp::csr_matrix<int, float, cusp::device_memory> A;
    cusp::csr_matrix<int, float, cusp::device_memory> B;
    //cusp::csr_matrix<int, int, cusp::device_memory> C;

    //cusp::io::read_matrix_market_file(A, argv[1]);
    cusp::io::read_matrix_market_file(B, argv[1]);

    coo A_COO,B_COO;
    csr A_CSR,B_CSR;
    A_COO.e = 0;
    A_COO.n = 0;
    readMTX(&A_COO, argv[1]);
    readMTX(&B_COO, argv[1]);
    sortCOO(A_COO, SORT_COO_ROW_BASE);

    sortCOO(B_COO, SORT_COO_ROW_BASE);
   
    printf("%d %d\n",A_COO.n,A_COO.e);

    A_CSR = transformCOOToCSR(A_COO);
    B_CSR = transformCOOToCSR(B_COO);

    int* CSR_CIDX_DEV;
    float* CSR_VAL_DEV;
    int* CSR_PTR_DEV;

    int* CSR_CIDX_DEV2;
    float* CSR_VAL_DEV2;
    int* CSR_PTR_DEV2;

    int* CSR_PTR_C;

    GpuTimer timer1, timer2;

    timer1.Start();
    cudaMalloc((void**) &(CSR_CIDX_DEV), sizeof(int)  *A_COO.e);    
    cudaMalloc((void**) &(CSR_PTR_DEV),  sizeof(int)  *(A_COO.n+1));
	cudaMemcpy((void*) (CSR_CIDX_DEV),	(const void*)(A_CSR.cidx),	sizeof(int)    *A_COO.e,	    cudaMemcpyHostToDevice);
    cudaMemcpy((void*) (CSR_PTR_DEV),	(const void*)(A_CSR.header),sizeof(int)    *(A_COO.n+1),    cudaMemcpyHostToDevice);
    
    cudaMalloc((void**) &(CSR_CIDX_DEV2), sizeof(int)  *A_COO.e);    
    cudaMalloc((void**) &(CSR_PTR_DEV2),  sizeof(int)  *(A_COO.n+1));
	cudaMemcpy((void*) (CSR_CIDX_DEV2),	(const void*)(B_CSR.cidx),	sizeof(int)    *A_COO.e,	    cudaMemcpyHostToDevice);
    cudaMemcpy((void*) (CSR_PTR_DEV2),	(const void*)(B_CSR.header),sizeof(int)    *(A_COO.n+1),    cudaMemcpyHostToDevice);
    
    cudaMalloc((void**) &(CSR_PTR_C), sizeof(int)  *A_COO.n);    
    cudaMemset(CSR_PTR_C, 0, sizeof(int)*A_COO.n);
    dim3 number_of_blocks(A_COO.n);
    dim3 size_of_block(32);
    calc_mem<<<number_of_blocks,size_of_block>>>(
                         CSR_CIDX_DEV , CSR_PTR_DEV,
                         CSR_CIDX_DEV2 , CSR_PTR_DEV2,
                         CSR_PTR_C);
    timer1.Stop();
    int* mem_required = (int*)malloc(sizeof(int)*A_COO.n);
    cudaMemcpy((void*) (mem_required),	(const void*)(CSR_PTR_C),sizeof(int)    *(A_COO.n),    cudaMemcpyDeviceToHost);
    int mem_total=0;
    for(int i=0;i<A_COO.n;i++){
        mem_total += mem_required[i];
    }
    cudaFree(CSR_CIDX_DEV);
    cudaFree(CSR_PTR_DEV);
    cudaFree(CSR_CIDX_DEV2);
    cudaFree(CSR_PTR_DEV2);
    cudaFree(CSR_PTR_C);
    printf("memtot %d\n",mem_total);

#define TRY 3
#define BLOCKSIZE 512

    int num_Blocks = ((A_COO.n - 1) / BLOCKSIZE) + 1;
    //cusp::csr_matrix<int, float, cusp::device_memory> C(A.num_rows, A.num_cols, mem_total);


    //cusp::csr_matrix<int, float, cusp::host_memory> D;
    
    cusp::constant_functor<float> zr;
    thrust::multiplies<float> combine;
    thrust::plus<float> reduce;


    long nnzC = 0;
    // allocate output matrix
    float t_kernel=0, t_d2h=0, t_write=0;
    float t1, t2, t3;
    struct timeval st, ed;
    for(int t=0;t<TRY;t++){ 
        t1=0; t2=0; t3=0;
        nnzC = 0;
        for (int b = 0; b < num_Blocks; ++b) {
            //printf("block %d / %d\n", b, num_Blocks);
            int start_Row = b * BLOCKSIZE;
            int end_Row = start_Row + BLOCKSIZE;
            if (b == num_Blocks - 1) end_Row = A_COO.n;
            int nr = end_Row - start_Row;
            int ne = A_CSR.header[end_Row] - A_CSR.header[start_Row];

            std::vector<int> rowPtr(nr + 1);
            std::vector<int> colInd(ne);
            std::vector<float> value(ne);

            rowPtr[0] = 0;
            int block_Mem = 0;
            for (int k = 0; k < nr + 1; ++k) {
                rowPtr[k] = A_CSR.header[start_Row + k] - A_CSR.header[start_Row];
                block_Mem += mem_required[start_Row + k];
            }
            memcpy(&colInd[0], A_CSR.cidx + A_CSR.header[start_Row], sizeof(int) * ne);
            memcpy(&value[0], A_CSR.val + A_CSR.header[start_Row], sizeof(float) * ne);
            //printf("%d %d %d\n", rowPtr.size(), colInd.size(), A_CSR.header[end_Row] - A_CSR.header[start_Row]);

            //cusp::csr_matrix<int, float, cusp::device_memory> block_A(nr, A.num_cols, rowPtr[nr]);
            cusp::csr_matrix<int, float, cusp::device_memory> A(nr, A_COO.n, ne);
            //A.resize(nr, A.num_cols, rowPtr[nr]);
            A.row_offsets = rowPtr;
            A.column_indices = colInd;
            A.values = value;


            cusp::csr_matrix<int, float, cusp::device_memory> C(nr, A.num_cols, block_Mem);
            //C.resize(nr, A.num_cols, block_Mem);

            gettimeofday(&st, NULL);
            cusp::multiply(A, B, C);
            //cusp::multiply(A, B, C, zr, combine, reduce);
            //cusp::generalized_spgemm(A, B, C, zr, combine, reduce);
            gettimeofday(&ed, NULL);
            t1 += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
            //printf("%f\n", (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec));

            cusp::csr_matrix<int, float, cusp::host_memory> D(C);
            cudaDeviceSynchronize();
            gettimeofday(&ed, NULL);
            t2 += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);

            filewrite(D.row_offsets.data(), D.column_indices.data(), D.values.data(), D.num_rows, D.num_entries);
            //printf("nr: %d, ne: %d\n", D.num_rows, D.num_entries);
            gettimeofday(&ed, NULL);
            t3 += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
            
            nnzC += D.num_entries;
        }
        printf("%f %f %f\n", t1, t2, t3);
        t_kernel += t1;
        t_d2h += t2;
        t_write += t3;
    }
    //printf("%s avg %lf  ",argv[1],t_spgemm/TRY);
    //std::cout<< "\npre mul : " << timer1.Elapsed()<<" Milli Second(s).\nmul : " << timer2.Elapsed() << " Milli Second(s).\n";
    std::cout<<"nnzC : "<<nnzC<<"\n";
    printf("%f\t%f\t%f\n", t_kernel/TRY, t_d2h/TRY, t_write/TRY);
    //cusp::print(C);

    return 0;
}
