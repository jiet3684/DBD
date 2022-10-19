#include "SSpMM.hpp"

int fd_Idx, fd_Val;
FILE *fp_Idx, *fp_Val;

void flush_Buffer1(int, int);
void flush_Buffer2(int, int);

#ifdef DEBUG
int write_Count = 0;
#endif

void* write_File(void *args) {
    struct timeval st, ed;

    int num_Blocks = ((A.nr - 1)/ ROWS_IN_BLOCK) + 1;
    //fd_Idx = open("Idx.output", O_CREAT | O_RDWR | O_TRUNC, 0644);
    //fd_Val = open("Val.output", O_CREAT | O_RDWR | O_TRUNC, 0644);
    fp_Idx = fopen("Idx.output", "wb");
    fp_Val = fopen("Val.output", "wb");

    pthread_t idx, val;

    int last_Block_Index = 0, current_Block_Index = 0;
    while (write_Args->finished == false) {
        __SYNC;
        while (write_Args->working == false) {
            if (write_Args->finished == true) {
                return NULL;
            }
            __SYNC;
        }

        current_Block_Index = write_Args->block_Index;

        gettimeofday(&st, NULL);
        if (write_Args->is_Buf1 == true) flush_Buffer1(last_Block_Index, current_Block_Index);
        else flush_Buffer2(last_Block_Index, current_Block_Index);
        //fflush(fp_Idx);
        //fflush(fp_Val);
        gettimeofday(&ed, NULL);
        //puts("End Writing\n");
	    elapsed_Time[4] += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
        write_Args->working = false;
#ifdef DEBUG
        printf("File Write: %d ~ %d\n", last_Block_Index, current_Block_Index);
#endif
        last_Block_Index = current_Block_Index;
    }
    
    //close(fd_Idx);
    //close(fd_Val);
    fclose(fp_Idx);
    fclose(fp_Val);

    return NULL;
}


void flush_Buffer1(int start_Block, int end_Block) {
    int block = start_Block;
    long int start_CPU = 0, end_CPU;
    long int start_GPU = 0, end_GPU;
    int buf_Info;
    //printf("BUF1. Start Block: %d, End Block: %d\n", start_Block, end_Block);
    
    while (block < end_Block) {
        //__SYNC;
        while (block_Queue[block].queue_Info == 0) __SYNC;

        buf_Info = block_Queue[block].queue_Info;
#ifdef DEBUG
        write_Count++;
#endif

        if (buf_Info == 1) {
            start_GPU = block_Queue[block].start_Loc;
            while (true) {
                block++;
                if (block == end_Block) break;
                while (block_Queue[block].queue_Info == 0) __SYNC;
                if (block_Queue[block].queue_Info != buf_Info) break;
            }
            end_GPU = block_Queue[block - 1].end_Loc;

            fwrite(gpu_BufIdx1 + start_GPU, sizeof(int), end_GPU - start_GPU, fp_Idx);
            fwrite(gpu_BufVal1 + start_GPU, sizeof(float), end_GPU - start_GPU, fp_Val);

#ifdef DEBUG
            if (end_GPU - start_GPU < 0) printf("\tBlock %d. Start: %ld, End: %ld\n", block - 1, start_GPU, end_GPU);
#endif
            start_GPU = end_GPU;
        }
        else {
            start_CPU = block_Queue[block].start_Loc;
            while (true) {
                block++;
                if (block == end_Block) break;
                while (block_Queue[block].queue_Info == 0) __SYNC;
                if (block_Queue[block].queue_Info != buf_Info) break;
            }
            end_CPU = block_Queue[block - 1].end_Loc;

            fwrite(cpu_BufIdx1 + start_CPU, sizeof(int), end_CPU - start_CPU, fp_Idx);
            fwrite(cpu_BufVal1 + start_CPU, sizeof(float), end_CPU - start_CPU, fp_Val);

#ifdef DEBUG
            if (end_CPU - start_CPU < 0) printf("\tBlock %d. Start: %ld, End: %ld\n", block - 1, start_CPU, end_CPU);
#endif
            start_CPU = end_CPU;
        }
    }
    //puts("BUF1 END.\n");
}

void flush_Buffer2(int start_Block, int end_Block) {
    int block = start_Block;
    long int start_CPU = 0, end_CPU;
    long int start_GPU = 0, end_GPU;
    int buf_Info;
    //printf("BUF2. Start Block: %d, End Block: %d\n", start_Block, end_Block);
    
    while (block < end_Block) {
        //__SYNC;
        while (block_Queue[block].queue_Info == 0) __SYNC;
#ifdef DEBUG
        write_Count++;
#endif

        buf_Info = block_Queue[block].queue_Info;

        if (buf_Info == 1) {
            start_GPU = block_Queue[block].start_Loc;
            while (true) {
                if (++block == end_Block) break;
                while (block_Queue[block].queue_Info == 0) __SYNC;
                if (block_Queue[block].queue_Info != buf_Info) break;
            }
            end_GPU = block_Queue[block - 1].end_Loc;

            fwrite(gpu_BufIdx2 + start_GPU, sizeof(int), end_GPU - start_GPU, fp_Idx);
            fwrite(gpu_BufVal2 + start_GPU, sizeof(float), end_GPU - start_GPU, fp_Val);

#ifdef DEBUG
            if (end_GPU - start_GPU < 0) printf("\tBlock %d. Start: %ld, End: %ld\n", block - 1, start_GPU, end_GPU);
#endif
            start_GPU = end_GPU;
        }
        else {
            start_CPU = block_Queue[block].start_Loc;
            while (true) {
                if (++block == end_Block) break;
                while (block_Queue[block].queue_Info == 0) __SYNC;
                if (block_Queue[block].queue_Info != buf_Info) break;
            }
            end_CPU = block_Queue[block - 1].end_Loc;

            fwrite(cpu_BufIdx2 + start_CPU, sizeof(int), end_CPU - start_CPU, fp_Idx);
            fwrite(cpu_BufVal2 + start_CPU, sizeof(float), end_CPU - start_CPU, fp_Val);

#ifdef DEBUG
            if (end_CPU - start_CPU < 0) printf("\tBlock %d. Start: %ld, End: %ld\n", block - 1, start_CPU, end_CPU);
#endif
            start_CPU = end_CPU;
        }
    }
    //puts("BUF2 END.\n");
}

/*void flush_Buffer() {
    long int byte_to_write = sizeof(int) * write_Args->buf_Len;
    printf("\tWrote %d MiBs to File\n", byte_to_write >> 20);
    long int file_Offset = lseek(fd_Idx, 0, SEEK_END);
    long int target_Offset = file_Offset + byte_to_write;
    ftruncate(fd_Idx, target_Offset);
    ftruncate(fd_Val, target_Offset);

    long int page_unit_diff = file_Offset - ((file_Offset >> 12) << 12);
    int *file_Idx = (int*)mmap(0, byte_to_write + page_unit_diff, PROT_WRITE, MAP_SHARED, fd_Idx, file_Offset - page_unit_diff);
    float *file_Val = (float*)mmap(0, byte_to_write + page_unit_diff, PROT_WRITE, MAP_SHARED, fd_Val, file_Offset - page_unit_diff);
    
    memcpy(file_Idx + (page_unit_diff / sizeof(int)), total_BufIdx, byte_to_write);
    memcpy(file_Val + (page_unit_diff / sizeof(float)), total_BufVal, byte_to_write);
    
    munmap(file_Idx, byte_to_write + page_unit_diff);
    munmap(file_Val, byte_to_write + page_unit_diff);

    printf("\tWrote %d MiBs to File\n\n", byte_to_write >> 20);
}*/

void write_Ptr() {
    struct timeval st, ed;
    gettimeofday(&st, NULL);

    /*FILE *fp_Ptr = fopen("Ptr.output", "wb");
    fwrite(ptrC, sizeof(int), A.nr + 1, fp_Ptr);
    fclose(fp_Ptr);*/
    int fd_Ptr = open("Ptr.output", O_CREAT | O_RDWR | O_TRUNC, 0644);
    long int byte_to_write = sizeof(int) * (A.nr + 1);
    ftruncate(fd_Ptr, byte_to_write);

    int *file_Ptr = (int*)mmap(0, byte_to_write, PROT_WRITE, MAP_SHARED, fd_Ptr, 0);

    memcpy(file_Ptr, ptrC, byte_to_write);

    munmap(file_Ptr, byte_to_write);

    close(fd_Ptr);
    gettimeofday(&ed, NULL);
    elapsed_Time[4] += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
}