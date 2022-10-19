#include "SSpMM.hpp"

storedBlock *block_Queue;

long int cpu_BufLoc = 0;
long int gpu_BufLoc = 0;

long int ne_CPU = 0;
long int ne_GPU = 0;

threadArgs *cpu_Args;
threadArgs *gpu_Args;
threadArgs *write_Args;

int *cpu_Queue;
int *gpu_Queue;

int calculate_Criteria(int);
inline int analyze_Block(int, int);
void change_Buffer(int);
int check_CPU(int);
int check_GPU(int);


void distribute_Workload() {
    struct timeval st, ed;

    int num_Blocks = ((A.nr - 1)/ ROWS_IN_BLOCK) + 1;
    block_Queue = (storedBlock*)malloc(sizeof(storedBlock) * num_Blocks);
    for (int b = 0; b < num_Blocks; ++b) block_Queue->queue_Info = 0;

    //puts(" - Analyze Blocks with Computation Amount -");
    int criteria = calculate_Criteria(num_Blocks);

#ifdef DEBUG
    int lower_than_warp = 0, comp_in_GPU = 0;
    for (int block = 0; block < num_Blocks; ++block) {
        int temp = analyze_Block(criteria, block_Info[block] / ROWS_IN_BLOCK);
        if (temp == 0) comp_in_GPU++;
        else if (temp == 1) lower_than_warp++;
    }
    printf("Top 10%% (Higher than %d): %d\n", criteria, comp_in_GPU);
    printf("Lower than Two Warp Size (64): %d\n\n", lower_than_warp);
#endif


    int cpu_ComputedBlocks = 0, gpu_ComputedBlocks = 0;
    //for (int iteration = 0; iteration < ITER; ++iteration) {
    cpu_Args = (threadArgs*)malloc(sizeof(threadArgs));
    gpu_Args = (threadArgs*)malloc(sizeof(threadArgs));
    *cpu_Args = {0, false, true, false};
    *gpu_Args = {0, false, true, false};
#ifndef COMPUTE
    write_Args = (threadArgs*)malloc(sizeof(writeArgs));
    *write_Args = {0, false, false, false};

    pthread_t write_Thread;
    pthread_create(&write_Thread, NULL, write_File, (void*)NULL);
#endif

    pthread_t cpu_Thread, gpu_Thread;
    pthread_create(&cpu_Thread, NULL, compute_CPU, (void*)NULL);
    pthread_create(&gpu_Thread, NULL, compute_GPU, (void*)NULL);

    int block_Index = 0;

    gettimeofday(&st, NULL);
#ifndef COMPUTE
    while (block_Index < num_Blocks) {
        __SYNC;
        // Add IF condition to check SSD whether is working or not. If not, change buffer and set write_Thread's Flag

        int analyzed_Result = analyze_Block(criteria, block_Info[block_Index] / ROWS_IN_BLOCK);
        //int analyzed_Result = 0;
        //printf("Block %d: Analyzed Result is %d\n", block_Index, analyzed_Result);
        if (analyzed_Result == 0) {
            int status = check_GPU(block_Index);
            if (status == 0) {   // compute with GPU immediately
                gpu_Args->block_Index = block_Index;
                gpu_Args->working = true;
                //block_Index++;
                gpu_ComputedBlocks++;
                ne_GPU += block_Info[block_Index++];
            }
            else if (status == 2) {
                while (write_Args->working == true) __SYNC;
                change_Buffer(block_Index);
            }
            else {
                while (gpu_Args->working == true) __SYNC;
                gpu_Args->block_Index = block_Index;
                gpu_Args->working = true;
                //block_Index++;
                gpu_ComputedBlocks++;
                ne_GPU += block_Info[block_Index++];
            }
            continue;
        }
        else if (analyzed_Result == 1) {
            int status = check_CPU(block_Index);
            if (status == 0) {   // compute with CPU
                cpu_Args->block_Index = block_Index;
                cpu_Args->working = true;
                //block_Index++;
                cpu_ComputedBlocks++;
                ne_CPU += block_Info[block_Index++];
                continue;
            }
            else if (status == 2) {
                while (write_Args->working == true) __SYNC;
                change_Buffer(block_Index);
                continue;
            }
        }
        while (true) {     // compute with any HW
            int status = check_GPU(block_Index);
            if (status == 0) {
                gpu_Args->block_Index = block_Index;
                gpu_Args->working = true;
                //block_Index++;
                gpu_ComputedBlocks++;
                ne_GPU += block_Info[block_Index++];
                break;
            }
            else if (status == 2) {
                while (write_Args->working == true) __SYNC;
                change_Buffer(block_Index);
            }
            status = check_CPU(block_Index);
            if (status == 0) {
                cpu_Args->block_Index = block_Index;
                cpu_Args->working = true;
                //block_Index++;
                cpu_ComputedBlocks++;
                ne_CPU += block_Info[block_Index++];
                break;
            }
            else if (status == 2) {
                while (write_Args->working == true) __SYNC;
                change_Buffer(block_Index);
            }
            __SYNC;
        }
    }
#else
    while (block_Index < num_Blocks) {
        __SYNC;
        int analyzed_Result = analyze_Block(criteria, block_Info[block_Index] / ROWS_IN_BLOCK);
        //int analyzed_Result = 0;
        if (analyzed_Result == 0) {
            while(gpu_Args->working == true) __SYNC;
            gpu_Args->block_Index = block_Index;
            gpu_Args->working = true;
            //block_Index++;
            gpu_ComputedBlocks++;
            ne_GPU += block_Info[block_Index++];
            continue;
        }
        else if (analyzed_Result == 1) {
            if (cpu_Args->working == false) {
                cpu_Args->block_Index = block_Index;
                cpu_Args->working = true;
                //block_Index++;
                cpu_ComputedBlocks++;
                ne_CPU += block_Info[block_Index++];
                continue;
            }
        }
        while (true) {     // compute with any HW
            if (gpu_Args->working == false) {
                gpu_Args->block_Index = block_Index;
                gpu_Args->working = true;
                //block_Index++;
                gpu_ComputedBlocks++;
                ne_GPU += block_Info[block_Index++];
                break;
            }
            if (cpu_Args->working == false) {
                cpu_Args->block_Index = block_Index;
                cpu_Args->working = true;
                //block_Index++;
                cpu_ComputedBlocks++;
                ne_CPU += block_Info[block_Index++];
                break;
            }
            __SYNC;
        }
    }
#endif
    while ((cpu_Args->working != false) || (gpu_Args->working != false)) __SYNC;
    cpu_Args->finished = true;
    gpu_Args->finished = true;

    pthread_join(cpu_Thread, NULL);
    pthread_join(gpu_Thread, NULL);

#ifndef COMPUTE
    ptrC[0] = 0;
#pragma unroll
    for (int row = 0; row < A.nr; ++row) {
        ptrC[row + 1] += ptrC[row];
    }
    while (write_Args->working != false) __SYNC;
    write_Args->block_Index = block_Index;
    write_Args->working = true;
    while (write_Args->working != false) __SYNC;
    write_Args->finished = true;
    pthread_join(write_Thread, NULL);
    write_Ptr();
    free(write_Args);
#endif
    gettimeofday(&ed, NULL);
    elapsed_Time[5] += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
    printf("%f\n", (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec));

    free(cpu_Args);
    free(gpu_Args);

    printf(" - Workload Distribution -\nCPUs\tBlocks:   %d (%.2f %)\n\tElements: %ld (%.2f %)\n", cpu_ComputedBlocks, (float)cpu_ComputedBlocks / (cpu_ComputedBlocks + gpu_ComputedBlocks) * 100, ne_CPU, (float)ne_CPU / (ne_CPU + ne_GPU) * 100);
    printf("GPUs\tBlocks:   %d (%.2f %)\n\tElements: %ld (%.2f %)\n\n", gpu_ComputedBlocks, (float)gpu_ComputedBlocks / (cpu_ComputedBlocks + gpu_ComputedBlocks) * 100, ne_GPU, (float)ne_GPU / (ne_CPU + ne_GPU) * 100);

    free(block_Queue);
}

int check_CPU(int block_Index) {
    int start_Row = block_Ptr[block_Index];
    int end_Row = block_Ptr[block_Index + 1];
    long int ne_in_this_block = h_Upp[end_Row] - h_Upp[start_Row];
    long int last_Location = cpu_BufLoc + ne_in_this_block;
    if (last_Location > ne_Buffer) return 2;

    if (cpu_Args->working == true) return 1;

    return 0;
}

int check_GPU(int block_Index) {
    int start_Row = block_Ptr[block_Index];
    int end_Row = block_Ptr[block_Index + 1];
    long int ne_in_this_block = h_Upp[end_Row] - h_Upp[start_Row];
    long int last_Location = gpu_BufLoc + ne_in_this_block;
    if (last_Location > ne_Buffer) return 2;

    if (gpu_Args->working == true) return 1;

    return 0;
}

int calculate_Criteria(int num_Blocks) {
    int avg_CompDensity = (neC / num_Blocks) / ROWS_IN_BLOCK;

    float stdDev = 0.0f;
    for (int block = 0; block < num_Blocks; ++block) {
        stdDev += (avg_CompDensity - (block_Info[block] / ROWS_IN_BLOCK)) * (avg_CompDensity - (block_Info[block] / ROWS_IN_BLOCK));
    }
    stdDev /= num_Blocks;
    stdDev = sqrt(stdDev);
    //printf("Standard Deviation: %f\n", stdDev);
    
    int top;
    top = (int)(1.28f * stdDev) + (int)avg_CompDensity; // 10%: 1.28    5%: 1.64

    return top;
}

inline int analyze_Block(int criteria, int avg_ComputationAmount) {
    if (avg_ComputationAmount > criteria) return 0;
    else if (avg_ComputationAmount <= 64) return 1;
    return 2;
}

void change_Buffer(int bid) {
    cpu_Args->is_Buf1 = !(cpu_Args->is_Buf1);
    gpu_Args->is_Buf1 = !(gpu_Args->is_Buf1);
    write_Args->is_Buf1 = !(write_Args->is_Buf1);
    cpu_BufLoc = 0;
    gpu_BufLoc = 0;

    write_Args->block_Index = bid;
    write_Args->working = true;
}