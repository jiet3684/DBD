/**
 * Arguments List
 * Executable File Name
 * CPU DRAM Size (GB, Optional)
 * GPU DRAM Size (GB, Optional)
 * Input File 1
 * Input File 2 (GB, Optional)
 * 
 * Number of Arguments
 * 2: Executable File Name, Input File 1.
 * 3: Executable File Name, Input File 1, Input File 2.
 * 4: Executable File Name, CPU DRAM Size, GPU DRAM Size, Input File 1.
 * 5: Executable File Name, CPU DRAM Size, GPU DRAM Size, Input File 1, Input File 2.
 * Add Block Size to Argument
 */

#include "SSpMM.hpp"

// ReadInput, Preprocessing, computeCPU, computeGPU, writeFile, Total
float elapsed_Time[7] = {0};

int cpu_MemBound, gpu_MemBound;
long int neC;

csr A, B;

int *h_Upp;

std::string input_FileA, input_FileB;

int *block_Ptr;
int *block_Info;

int *h_IdxC = NULL;
float *h_ValC = NULL;

// Buffer which stores computed result from CPU, GPU
int *ptrC = NULL;

int *cpu_BufIdx1 = NULL, *gpu_BufIdx1 = NULL;
float *cpu_BufVal1 = NULL, *gpu_BufVal1 = NULL;

int *cpu_BufIdx2 = NULL, *gpu_BufIdx2 = NULL;
float *cpu_BufVal2 = NULL, *gpu_BufVal2 = NULL;

int *cpu_Ptr_for_Collecting = NULL;

float *cpu_Dense = NULL;

// Total number of elements that can be stored to buffer
long int ne_Buffer;

void initialize(int argc, char **argv);
void read_InputB();
int divide_Blocks();   // Divide Matrix A into blocks of appropriate size, returns number of blocks
void free_Resource();


int main(int argc, char **argv) {
    puts("\nLarge-Scaled Sparse Matrix - Sparse Matrix Multiplication. (L3SpMM)\n");

    remove("Ptr.output");
    remove("Idx.output");
    remove("Val.output");

    /*int pid = fork();
    if (pid == 0) execlp("sh", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches", NULL);
    else wait(NULL);*/

    initialize(argc, argv);

    //struct timeval total_Start, total_End;
    //gettimeofday(&total_Start, NULL);

    read_CSR(input_FileA, &A);
    read_CSR(input_FileB, &B);
    
    initialize_A();
    initialize_B();

    //elapsed_Time[0] = (float)(temp_End.tv_sec - temp_Start.tv_sec) + 0.000001 * (float)(temp_End.tv_usec - temp_Start.tv_usec);

    //gettimeofday(&temp_Start, NULL);

    int num_Blocks = divide_Blocks();
    if (num_Blocks == -1) {
        free_Resource();
        return 0;
    }
    //gettimeofday(&temp_End, NULL);
    //elapsed_Time[1] = (float)(temp_End.tv_sec - temp_Start.tv_sec) + 0.000001 * (float)(temp_End.tv_usec - temp_Start.tv_usec);

    for (int iter = 0; iter < ITER; ++iter)
        distribute_Workload();

    //gettimeofday(&total_End, NULL);
    //elapsed_Time[6] = (float)(total_End.tv_sec - total_Start.tv_sec) + 0.000001 * (float)(total_End.tv_usec - total_Start.tv_usec);

    //printf("-------------- TIME --------------\nRead & Convert\t\t%f s\n", elapsed_Time[0]);
    printf("------------ AVG TIME ------------\n");
    printf("Preprocessing\t\t%f s\n", elapsed_Time[1]);
    printf("Compute CPU\t\t%f s\n", elapsed_Time[2] / ITER);
    printf("Compute GPU\t\t%f s\n", elapsed_Time[3] / ITER);
    printf("Write File\t\t%f s\n", elapsed_Time[4] / ITER);
    printf("Kernel Execution\t%f s\n", elapsed_Time[5] / ITER);
    printf("Total Execution\t\t%f s\n", (elapsed_Time[5] / ITER) + elapsed_Time[1]);
    //printf("Total Execution\t\t%f s\n\n", elapsed_Time[6]);
    //printf("GFLOPS: %f\n\n", (float)neC / elapsed_Time[5]);

#ifdef DEBUG
    long ne_Total = ne_CPU + ne_GPU;
    puts("\n - DEBUG -");
    printf("Merged %d Blocks while Writing (%f)\n",  num_Blocks - (write_Count / ITER), (float)(num_Blocks - (write_Count / ITER)) / num_Blocks);
    printf("Computed Total %ld Elements (%ld Bytes = %d MebiBytes)\n", neC, neC << 2, neC >> 18);
    printf("Actual Total %ld Elements (%ld Bytes = %d MebiBytes)\n", ptrC[A.nr], ptrC[A.nr] << 2, ptrC[A.nr] >> 18);
    printf("Reduced to %.2f %% After Merge\n\n", (float)ptrC[A.nr] / ne_Total * 100);
#endif

    free_Resource();
}


void free_Resource() {
    if (A.ptr) free(A.ptr);
    if (A.idx) free(A.idx);
    if (A.val) free(A.val);

    if (B.ptr) free(B.ptr);
    if (B.idx) free(B.idx);
    if (B.val) free(B.val);

    if (block_Ptr) free(block_Ptr);
    if (block_Info) free(block_Info);
    if (h_Upp) free(h_Upp);

    if (h_IdxC) free(h_IdxC);
    if (h_ValC) free(h_ValC);
    if (cpu_Dense) free(cpu_Dense);

#ifndef COMPUTE
    if (cpu_BufIdx1) free(cpu_BufIdx1);
    if (cpu_BufVal1) free(cpu_BufVal1);
    if (cpu_BufIdx2) free(cpu_BufIdx2);
    if (cpu_BufVal2) free(cpu_BufVal2);

    if (cpu_Ptr_for_Collecting) free(cpu_Ptr_for_Collecting);
#endif

    free_GPU();
}

void initialize(int argc, char **argv) {
    if (argc == 2) {
        cpu_MemBound = 28 << 10;
        gpu_MemBound = 22 << 10;
        input_FileA = argv[1];
        input_FileB = argv[1];
    }
    else if (argc == 3) {
        cpu_MemBound = 28 << 10;
        gpu_MemBound = 22 << 10;
        input_FileA = argv[1];
        input_FileB = argv[2];
    }
    else if (argc == 4) {
        cpu_MemBound = atoi(argv[1]) << 10;
        gpu_MemBound = atoi(argv[2]) << 10;
        input_FileA = argv[3];
        input_FileB = argv[3];
    }
    else if (argc == 5) {
        cpu_MemBound = atoi(argv[1]) << 10;
        gpu_MemBound = atoi(argv[2]) << 10;
        input_FileA = argv[3];
        input_FileB = argv[4];
    }
    else {
        puts("Invalid Arguments.");
        puts("Usage:\t<1> Executable File Name, Input File 1");
        puts("\t<2> Executable File Name, Input File 1, Input File 2");
        puts("\t<3> Executable File Name, CPU DRAM Size, GPU DRAM Size, Input File 1");
        puts("\t<4> Executable File Name, CPU DRAM Size, GPU DRAM Size, Input File 1, Input File 2\n");
        exit(-1);
    }

    initialize_GPU();
}

#ifndef COMPUTE
int make_Buffer(int max_CalulationAmount) {
    //puts("Allocate Buffer for Calculation Result from CPU & GPU.");

    long int cpu_MemUsage = 0, gpu_MemUsage = 0;
    cpu_MemUsage += A.nr * sizeof(int) + A.ne * sizeof(int) + A.ne * sizeof(float);    // PTR_A, IDX_A, VAL_A
    cpu_MemUsage += B.nr * sizeof(int) + B.ne * sizeof(int) + B.ne * sizeof(float);
    cpu_MemUsage += 2 * A.nr * sizeof(int);  // UPPER_BOUND, PTR_C
    cpu_MemUsage += CPU_DENSE * A.nc * sizeof(float);    // CPU_DENSE
    cpu_MemUsage += max_CalulationAmount * (sizeof(int) + sizeof(float));
    cpu_MemUsage >>= 20;
    printf("CPU Memory Usage: %ld MiBs\n", cpu_MemUsage);
    
    gpu_MemUsage += 2 * A.nr * sizeof(int) + A.ne * sizeof(int) + A.ne * sizeof(float);
    gpu_MemUsage += B.nr * sizeof(int) + B.ne * sizeof(int) + B.ne * sizeof(float);
    gpu_MemUsage += 2 * ((ROWS_IN_BLOCK + 1) * sizeof(int) + 2 * max_CalulationAmount * (sizeof(int) + sizeof(float)));  // Stores output of one block
    gpu_MemUsage += sizeof(float) * (ROWS_IN_BLOCK) * A.nc;  // gpu_Dense array for merge
    gpu_MemUsage >>= 20;
    printf("GPU Memory Usage: %ld MiBs\n\n", gpu_MemUsage);

    int cpu_FreeMem = cpu_MemBound - (int)cpu_MemUsage;
    int gpu_FreeMem = gpu_MemBound - (int)gpu_MemUsage;
    if (cpu_FreeMem <= 0 || gpu_FreeMem <= 0) 
    {
        fprintf(stderr, "Not Enough Memory for Dense Matrix for Result.\n");
        return -1;
    }

    long block_MaxUsage = ((long)max_CalulationAmount * (sizeof(int) + sizeof(float)) * 4) >> 20;
    int multi_Blocks;
    for (multi_Blocks = 32; multi_Blocks > 0; --multi_Blocks) {
        int now_Usage = block_MaxUsage * multi_Blocks;
        if (now_Usage <= cpu_FreeMem) break;
    }
    if (multi_Blocks == 0) {
        fprintf(stderr, "Not Enough Memory for Buffer.\n");
        return -1;
    }

    cpu_Ptr_for_Collecting = (int*)malloc(sizeof(int) * (ROWS_IN_BLOCK + 1));
    h_IdxC = (int*)malloc(sizeof(int) * max_CalulationAmount);
    h_ValC = (float*)malloc(sizeof(float) * max_CalulationAmount);

    cpu_Dense = (float*)malloc(sizeof(float) * A.nc * CPU_DENSE);
    memset(cpu_Dense, 0, sizeof(float) * A.nc * CPU_DENSE);

    ne_Buffer = (long)max_CalulationAmount * multi_Blocks;
    printf("Buffer Size: %d MiBs (%d x %d)\n", ne_Buffer >> 18, max_CalulationAmount, multi_Blocks);

    cpu_BufIdx1 = (int*)malloc(sizeof(int) * ne_Buffer);
    cpu_BufVal1 = (float*)malloc(sizeof(float) * ne_Buffer);
    if ((cpu_BufIdx1 == NULL) || (cpu_BufVal1 == NULL)) {
        fprintf(stderr, "Cannot Allocate Buffers for Result from CPUs.\n");
        return -1;
    }

    cpu_BufIdx2 = (int*)malloc(sizeof(int) * ne_Buffer);
    cpu_BufVal2 = (float*)malloc(sizeof(float) * ne_Buffer);
    if ((cpu_BufIdx2 == NULL) || (cpu_BufVal2 == NULL)) {
        fprintf(stderr, "Cannot Allocate Buffers for Result from CPUs.\n");
        return -1;
    }

    cpu_MemUsage += (ne_Buffer * (sizeof(int) + sizeof(float)) * 2) >> 20;
    printf("CPU Memory Usage: %ld MiBs (After Buffer Allocation)\n\n", cpu_MemUsage);

    return initialize_C(max_CalulationAmount, multi_Blocks);
}
#endif

int divide_Blocks() {
    struct timeval st, ed;
    gettimeofday(&st, NULL);

    get_UpperBound();

    int num_Blocks = ((A.nr - 1)/ ROWS_IN_BLOCK) + 1;
    block_Ptr = (int*)malloc(sizeof(int) * (num_Blocks + 1));
    block_Ptr[0] = 0;
    block_Info = (int*)malloc(sizeof(int) * num_Blocks);

    printf("Divide Matrix A into %d Blocks (Block Size: %d)\n\n", num_Blocks, ROWS_IN_BLOCK);

    int row_Offset = 0;
    int acc_NNZs = 0;
    int block_Index = 0;

    while (block_Index < num_Blocks - 1) {
        int next_Row_Offset = row_Offset + ROWS_IN_BLOCK;
        block_Ptr[block_Index + 1] = next_Row_Offset;
        block_Info[block_Index] = (h_Upp[next_Row_Offset] - h_Upp[row_Offset]);
        if (block_Info[block_Index] < 0) {
            puts("NE in Block Exceeds INTEGER Bound.");
            return -1;
        }

        row_Offset = next_Row_Offset;
        block_Index++;
    }
    block_Ptr[num_Blocks] = A.nr;
    block_Info[num_Blocks - 1] = h_Upp[A.nr] - h_Upp[block_Ptr[num_Blocks - 1]];

    int max_CalulationAmount = 0;
    for (int block = 0; block < num_Blocks; ++block) {
        if (block_Info[block] > max_CalulationAmount) max_CalulationAmount = block_Info[block];
        neC += block_Info[block];
        //block_Info[block] /= ROWS_IN_BLOCK;
    }
    //printf("max neC: %d\n", max_CalulationAmount);
    gettimeofday(&ed, NULL);
    elapsed_Time[1] += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);

#ifndef COMPUTE
    if (make_Buffer(max_CalulationAmount) == -1) {
        return -1;
    }
#else
    if (initialize_C(max_CalulationAmount) == -1)   return -1;
#endif

    return num_Blocks;
}



