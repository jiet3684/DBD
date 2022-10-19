//////////////////////////////////////////////////////////////////////////
// < A CUDA/OpenCL General Sparse Matrix-Matrix Multiplication Program >
//
// < See paper:
// Weifeng Liu and Brian Vinter, "An Efficient GPU General Sparse
// Matrix-Matrix Multiplication for Irregular Data," Parallel and
// Distributed Processing Symposium, 2014 IEEE 28th International
// (IPDPS '14), pp.370-381, 19-23 May 2014
// for details. >
//////////////////////////////////////////////////////////////////////////

#ifndef BHSPARSE_H
#define BHSPARSE_H

#include "bhsparse_cuda.h"

class bhsparse
{
public:
    bhsparse();
    int initPlatform(bool *spgemm_platform);
    int initData(int m, int k, int n,
             int nnzA, value_type *csrValA, index_type *csrRowPtrA, index_type *csrColIndA,
             int nnzB, value_type *csrValB, index_type *csrRowPtrB, index_type *csrColIndB,
             index_type *csrRowPtrC);
    int spgemm();
    int warmup();

    int get_nnzC();
    int get_C(index_type *csrColIndC, value_type *csrValC);

    int freePlatform();
    int free_mem();

private:
    bool *_spgemm_platform;

    bhsparse_cuda   *_bh_sparse_cuda;

    StopWatchInterface *_stage1_timer;
    StopWatchInterface *_stage2_timer;
    StopWatchInterface *_stage3_timer;
    StopWatchInterface *_stage4_timer;

    int spgemm_cuda();

    int statistics();

    int compute_nnzC_Ct_cuda();

    int copy_Ct_to_C_cuda();

    int _m;
    int _k;
    int _n;

    size_t _nnzCt_full;

    // A
    int    _nnzA;
    int   *_h_csrRowPtrA;
    int   *_h_csrColIndA;
    value_type *_h_csrValA;

    // B
    int    _nnzB;
    int   *_h_csrRowPtrB;
    int   *_h_csrColIndB;
    value_type *_h_csrValB;

    // C
    int    _nnzC;
    int   *_h_csrRowPtrC;
    int   *_h_csrColIndC;
    value_type *_h_csrValC;

    // Ct
    //int    _nnzCt;
    int   *_h_csrRowPtrCt;
    //int   *_h_csrColIndCt;
    //value_type *_h_csrValCt;

    // statistics
    int   *_h_counter;
    int   *_h_counter_one;
    int   *_h_counter_sum;
    int   *_h_queue_one;

};

bhsparse::bhsparse()
{

}

int bhsparse::initPlatform(bool *spgemm_platform)
{
    int err = 0;

    _spgemm_platform = spgemm_platform;

    for (int i = 0; i < NUM_PLATFORMS; i++)
    {
        if (_spgemm_platform[i])
        {
            switch (i)
            {
            case BHSPARSE_CUDA:
            {
                _bh_sparse_cuda = new bhsparse_cuda();
                err = _bh_sparse_cuda->initPlatform();
                break;
            }
            case BHSPARSE_OPENCL:
            {
                //_bh_sparse_opencl = new bhsparse_opencl();
                //err = _bh_sparse_opencl->initPlatform();
                break;
            }
            }
        }
    }

    return err;
}

int bhsparse::freePlatform()
{
    int err = 0;

    for (int i = 0; i < NUM_PLATFORMS; i++)
    {
        if (_spgemm_platform[i])
        {
            switch (i)
            {
            case BHSPARSE_CUDA:
                err = _bh_sparse_cuda->freePlatform();
                break;
            case BHSPARSE_OPENCL:
                //err = _bh_sparse_opencl->freePlatform();
                break;
            }
        }
    }

    return err;
}

int bhsparse::free_mem()
{
    int err = 0;

    free(_h_counter);
    free(_h_counter_one);
    free(_h_counter_sum);
    free(_h_queue_one);
    free(_h_csrRowPtrCt);

    for (int i = 0; i < NUM_PLATFORMS; i++)
    {
        if (_spgemm_platform[i])
        {
            switch (i)
            {
            case BHSPARSE_CUDA:
                err = _bh_sparse_cuda->free_mem();
                break;
            case BHSPARSE_OPENCL:
                //err = _bh_sparse_opencl->free_mem();
                break;
            }
        }
    }

    return err;
}


int bhsparse::initData(int m, int k, int n,
                   int nnzA, value_type *csrValA, index_type *csrRowPtrA, index_type *csrColIndA,
                   int nnzB, value_type *csrValB, index_type *csrRowPtrB, index_type *csrColIndB,
                   index_type *csrRowPtrC)
{
    int err = 0;

    _stage1_timer = NULL;
    _stage2_timer = NULL;
    _stage3_timer = NULL;
    _stage4_timer = NULL;

    sdkCreateTimer(&_stage1_timer);
    sdkCreateTimer(&_stage2_timer);
    sdkCreateTimer(&_stage3_timer);
    sdkCreateTimer(&_stage4_timer);

    _m = m;
    _k = k;
    _n = n;

    _nnzA = nnzA;
    _nnzB = nnzB;
    _nnzC = 0;

    // A
    _h_csrRowPtrA = csrRowPtrA;
    _h_csrColIndA = csrColIndA;
    _h_csrValA    = csrValA;

    // B
    _h_csrRowPtrB = csrRowPtrB;
    _h_csrColIndB = csrColIndB;
    _h_csrValB    = csrValB;

    // C
    _h_csrRowPtrC = csrRowPtrC;

    // Ct
    _h_csrRowPtrCt = (index_type *)  malloc((_m+1) * sizeof(index_type));
    memset(_h_csrRowPtrCt, 0, (_m+1) * sizeof(index_type));

    // statistics
    _h_counter = (int *)malloc(NUM_SEGMENTS * sizeof(int));
    memset(_h_counter, 0, NUM_SEGMENTS * sizeof(int));

    _h_counter_one = (int *)malloc((NUM_SEGMENTS + 1) * sizeof(int));
    memset(_h_counter_one, 0, (NUM_SEGMENTS + 1) * sizeof(int));

    _h_counter_sum = (int *)malloc((NUM_SEGMENTS + 1) * sizeof(int));
    memset(_h_counter_sum, 0, (NUM_SEGMENTS + 1) * sizeof(int));

    _h_queue_one = (int *)  malloc(TUPLE_QUEUE * _m * sizeof(int));
    memset(_h_queue_one, 0, TUPLE_QUEUE * _m * sizeof(int));

    for (int i = 0; i < NUM_PLATFORMS; i++)
    {
        if (_spgemm_platform[i])
        {
            switch (i)
            {
            case BHSPARSE_CUDA:
                err = _bh_sparse_cuda->initData(_m, _k, _n,
                                      _nnzA, _h_csrValA, _h_csrRowPtrA, _h_csrColIndA,
                                      _nnzB, _h_csrValB, _h_csrRowPtrB, _h_csrColIndB,
                                      _h_csrRowPtrC, _h_csrRowPtrCt, _h_queue_one);
                break;
            case BHSPARSE_OPENCL:
                //err = _bh_sparse_opencl->initData(_m, _k, _n,
                //                        _nnzA, _h_csrValA, _h_csrRowPtrA, _h_csrColIndA,
                //                        _nnzB, _h_csrValB, _h_csrRowPtrB, _h_csrColIndB,
                //                        _h_csrRowPtrC, _h_csrRowPtrCt, _h_queue_one);
                break;
            }
        }
    }

    return err;
}

int bhsparse::spgemm()
{
    int err = 0;

    for (int i = 0; i < NUM_PLATFORMS; i++)
    {
        if (_spgemm_platform[i])
        {
            StopWatchInterface *timer = NULL;
            sdkCreateTimer(&timer);

            sdkStartTimer(&timer);

            switch (i)
            {
            case BHSPARSE_CUDA:
                err = spgemm_cuda();
                //cout << "[ CUDA ]";
                break;
            case BHSPARSE_OPENCL:
                //err = spgemm_opencl();
                //cout << "[ OpenCL ]";
                break;
            }

            sdkStopTimer(&timer);
            double time = sdkGetTimerValue(&timer);
            //cout << " SpGEMM time: " << time << " ms. Gflops = " << 2.0 * (double)_nnzCt_full / (time * 1.0e+6) << endl;
        }
    }
    if(err != BHSPARSE_SUCCESS) { cout << "spgemm error = " << err << endl; return err; }

    return err;
}

int bhsparse::spgemm_cuda()
{
    int err = 0;

    // STAGE 1 : compute nnzCt
    sdkStartTimer(&_stage1_timer);
    err  = _bh_sparse_cuda->compute_nnzCt();
    err |= _bh_sparse_cuda->kernel_barrier();
    if(err != BHSPARSE_SUCCESS) { cout << "compute_nnzCt error = " << err << endl; return err; }
    sdkStopTimer(&_stage1_timer);
    cout << "STAGE 1 time: " << sdkGetTimerValue(&_stage1_timer) << " ms." << endl;

    // STAGE 2 - STEP 1 : statistics
    sdkStartTimer(&_stage2_timer);
    int nnzCt = statistics();

    // STAGE 2 - STEP 2 : create Ct
    err = _bh_sparse_cuda->create_Ct(nnzCt);
    if(err != BHSPARSE_SUCCESS) { cout << "create Ct error = " << err << endl; return err; }
    sdkStopTimer(&_stage2_timer);
    cout << "STAGE 2 time: " << sdkGetTimerValue(&_stage2_timer) << " ms." << endl;

    // STAGE 3 - STEP 1 : compute nnzC and Ct
    sdkStartTimer(&_stage3_timer);
    err = compute_nnzC_Ct_cuda();
    if(err != BHSPARSE_SUCCESS) { cout << "compute_C error = " << err << endl; return err; }

    // STAGE 3 - STEP 2 : malloc C on devices
    err  = _bh_sparse_cuda->create_C();
    if(err != BHSPARSE_SUCCESS) { cout << "create_C error = " << err << endl; return err; }
    sdkStopTimer(&_stage3_timer);
    cout << "STAGE 3 time: " << sdkGetTimerValue(&_stage3_timer) / 100 << " s." << endl;

    // STAGE 4 : copy Ct to C
    sdkStartTimer(&_stage4_timer);
    err  = copy_Ct_to_C_cuda();
    err |= _bh_sparse_cuda->kernel_barrier();
    if(err != BHSPARSE_SUCCESS) { cout << "copy_Ct_to_C error = " << err << endl; return err; }
    sdkStopTimer(&_stage4_timer);
    cout << "STAGE 4 time: " << sdkGetTimerValue(&_stage4_timer) << " ms." << endl;

    cout << "pre/postprocessing time: " << (sdkGetTimerValue(&_stage1_timer) + sdkGetTimerValue(&_stage2_timer) + sdkGetTimerValue(&_stage4_timer)) / 100 << endl;

    return err;
}

int bhsparse::warmup()
{
    int err = 0;

    for (int i = 0; i < NUM_PLATFORMS; i++)
    {
        if (_spgemm_platform[i])
        {
            switch (i)
            {
            case BHSPARSE_CUDA:
                err = _bh_sparse_cuda->warmup();
                break;
            case BHSPARSE_OPENCL:
                //err = _bh_sparse_opencl->warmup();
                break;
            }
        }
    }
    if(err != BHSPARSE_SUCCESS) { cout << "warmup error = " << err << endl; return err; }

    return err;
}

int bhsparse::statistics()
{
    int nnzCt = 0;
    _nnzCt_full = 0;

    // statistics for queues
    int count, position;

    for (int i = 0; i < _m; i++)
    {
        count = _h_csrRowPtrCt[i];

        if (count >= 0 && count <= 121)
        {
            _h_counter_one[count]++;
            _h_counter_sum[count] += count;
            _nnzCt_full += count;
        }
        else if (count >= 122 && count <= 128)
        {
            _h_counter_one[122]++;
            _h_counter_sum[122] += count;
            _nnzCt_full += count;
        }
        else if (count >= 129 && count <= 256)
        {
            _h_counter_one[123]++;
            _h_counter_sum[123] += count;
            _nnzCt_full += count;
        }
        else if (count >= 257 && count <= 512)
        {
            _h_counter_one[124]++;
            _h_counter_sum[124] += count;
            _nnzCt_full += count;
        }
        else if (count >= 513)
        {
            _h_counter_one[127]++;
            _h_counter_sum[127] += MERGELIST_INITSIZE;
            _nnzCt_full += count;
        }
    }

    // exclusive scan

    int old_val, new_val;

    old_val = _h_counter_one[0];
    _h_counter_one[0] = 0;
    for (int i = 1; i <= NUM_SEGMENTS; i++)
    {
        new_val = _h_counter_one[i];
        _h_counter_one[i] = old_val + _h_counter_one[i-1];
        old_val = new_val;
    }

    old_val = _h_counter_sum[0];
    _h_counter_sum[0] = 0;
    for (int i = 1; i <= NUM_SEGMENTS; i++)
    {
        new_val = _h_counter_sum[i];
        _h_counter_sum[i] = old_val + _h_counter_sum[i-1];
        old_val = new_val;
    }

    nnzCt = _h_counter_sum[NUM_SEGMENTS];
    //cout << "allocated size " << nnzCt << " out of full size " << _nnzCt_full << endl;

    for (int i = 0; i < _m; i++)
    {
        count = _h_csrRowPtrCt[i];

        if (count >= 0 && count <= 121)
        {
            position = _h_counter_one[count] + _h_counter[count];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[count];
            _h_counter_sum[count] += count;
            _h_counter[count]++;
        }
        else if (count >= 122 && count <= 128)
        {
            position = _h_counter_one[122] + _h_counter[122];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[122];
            _h_counter_sum[122] += count;
            _h_counter[122]++;
        }
        else if (count >= 129 && count <= 256)
        {
            position = _h_counter_one[123] + _h_counter[123];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[123];
            _h_counter_sum[123] += count;
            _h_counter[123]++;
        }
        else if (count >= 257 && count <= 512)
        {
            position = _h_counter_one[124] + _h_counter[124];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[124];
            _h_counter_sum[124] += count;
            _h_counter[124]++;
        }
        else if (count >= 513)
        {
            position = _h_counter_one[127] + _h_counter[127];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[127];
            _h_counter_sum[127] += MERGELIST_INITSIZE;
            _h_counter[127]++;
        }
    }

    return nnzCt;
}

int bhsparse::compute_nnzC_Ct_cuda()
{
    int err = 0;
    int counter = 0;

    for (int j = 0; j < NUM_SEGMENTS; j++)
    {
        counter = _h_counter_one[j+1] - _h_counter_one[j];
        if (counter != 0)
        {
            if (j == 0)
            {
                int num_threads = GROUPSIZE_256;
                int num_blocks  = ceil((double)counter / (double)num_threads);
                err = _bh_sparse_cuda->compute_nnzC_Ct_0(num_threads, num_blocks, j, counter, _h_counter_one[j]);
            }
            else if (j == 1)
            {
                int num_threads = GROUPSIZE_256;
                int num_blocks  = ceil((double)counter / (double)num_threads);
                err = _bh_sparse_cuda->compute_nnzC_Ct_1(num_threads, num_blocks, j, counter, _h_counter_one[j]);
            }
            else if (j > 1 && j <= 32)
            {
                int num_threads = WARPSIZE_NV_2HEAP; //_warpsize;
                int num_blocks = ceil((double)counter / (double)num_threads);
                err = _bh_sparse_cuda->compute_nnzC_Ct_2heap_noncoalesced(num_threads, num_blocks, j, counter, _h_counter_one[j]);
            }
            else if (j > 32 && j <= 64)
            {
                int num_threads = 32;
                int num_blocks  = counter;
                err = _bh_sparse_cuda->compute_nnzC_Ct_bitonic(num_threads, num_blocks, j, _h_counter_one[j]);
            }
            else if (j > 64 && j <= 122)
            {
                int num_threads = 64;
                int num_blocks  = counter;
                err = _bh_sparse_cuda->compute_nnzC_Ct_bitonic(num_threads, num_blocks, j, _h_counter_one[j]);
            }
            else if (j == 123)
            {
                int num_threads = 128;
                int num_blocks  = counter;
                err = _bh_sparse_cuda->compute_nnzC_Ct_bitonic(num_threads, num_blocks, j, _h_counter_one[j]);
            }
            else if (j == 124)
            {
                int num_threads = 256;
                int num_blocks  = counter;
                err = _bh_sparse_cuda->compute_nnzC_Ct_bitonic(num_threads, num_blocks, j, _h_counter_one[j]);
            }
            else if (j == 127)
            {
                int count_next = counter;
                int num_threads, num_blocks, mergebuffer_size;

                int num_threads_queue [5]      = {64,  128,  256,  256, 256};
                int mergebuffer_size_queue [5] = {256, 512, 1024, 2048, 2304}; //{256, 464, 924, 1888, 3840};

                int queue_counter = 0;

                while (count_next > 0)
                {
                    num_blocks  = count_next;

                    if (queue_counter < 5)
                    {
                        num_threads = num_threads_queue[queue_counter];
                        mergebuffer_size = mergebuffer_size_queue[queue_counter];

                        err = _bh_sparse_cuda->compute_nnzC_Ct_mergepath(num_threads, num_blocks, j, mergebuffer_size,
                                                                         _h_counter_one[j], &count_next, MERGEPATH_LOCAL);
                        if(err != BHSPARSE_SUCCESS) { cout << "compute_C_merge_local error = " << err << endl; return err; }

                        queue_counter++;
                    }
                    else
                    {
                        num_threads = num_threads_queue[4];
                        mergebuffer_size += mergebuffer_size_queue[4];

                        err = _bh_sparse_cuda->compute_nnzC_Ct_mergepath(num_threads, num_blocks, j, mergebuffer_size,
                                                                           _h_counter_one[j], &count_next, MERGEPATH_GLOBAL);
                        if(err != BHSPARSE_SUCCESS) { cout << "compute_C_merge_global error = " << err << endl; return err; }
                    }

//                    if (mergebuffer_size > 25600)
//                    {
//                        cout << "mergebuffer_size = " << mergebuffer_size << " is too large. break." << endl;
//                        break;
//                    }
                }

                if (count_next != 0)
                    cout << "Remaining = " << count_next << endl;
            }

            if(err != BHSPARSE_SUCCESS) { cout << "compute_C_2heap_bitonic error = " << err << endl; return err; }
        }
    }

    return err;
}

int bhsparse::copy_Ct_to_C_cuda()
{
    int err = 0;
    int counter = 0;

    // start from 1 instead of 0, since empty queues do not need this copy
    for (int j = 1; j < NUM_SEGMENTS; j++)
    {
        counter = _h_counter_one[j+1] - _h_counter_one[j];
        if (counter != 0)
        {
            if (j == 1)
            {
                int num_threads = GROUPSIZE_256;
                int num_blocks  = ceil((double)counter / (double)num_threads);
                err = _bh_sparse_cuda->copy_Ct_to_C_Single( num_threads, num_blocks, counter, _h_counter_one[j]);
            }
            else if (j > 1 && j <= 32)
                err = _bh_sparse_cuda->copy_Ct_to_C_Loopless(   32, counter, j, _h_counter_one[j]);
            else if (j > 32 && j <= 64)
                err = _bh_sparse_cuda->copy_Ct_to_C_Loopless(   64, counter, j, _h_counter_one[j]);
            else if (j > 64 && j <= 96)
                err = _bh_sparse_cuda->copy_Ct_to_C_Loopless(   96, counter, j, _h_counter_one[j]);
            else if (j > 96 && j <= 122)
                err = _bh_sparse_cuda->copy_Ct_to_C_Loopless(  128, counter, j, _h_counter_one[j]);
            else if (j == 123)
                err = _bh_sparse_cuda->copy_Ct_to_C_Loopless(  256, counter, j, _h_counter_one[j]);
            else if (j == 124)
                err = _bh_sparse_cuda->copy_Ct_to_C_Loopless(  512, counter, j, _h_counter_one[j]);
            else if (j == 127)
                err = _bh_sparse_cuda->copy_Ct_to_C_Loop( 256, counter, j, _h_counter_one[j]);

            if(err != BHSPARSE_SUCCESS) return err;
        }
    }

    return err;
}

int bhsparse::get_nnzC()
{
    int nnzC;

    for (int i = 0; i < NUM_PLATFORMS; i++)
    {
        if (_spgemm_platform[i])
        {
            switch (i)
            {
            case BHSPARSE_CUDA:
                nnzC = _bh_sparse_cuda->get_nnzC();
                break;
            case BHSPARSE_OPENCL:
                //nnzC = _bh_sparse_opencl->get_nnzC();
                break;
            }
        }
    }

    return nnzC;
}

int bhsparse::get_C(index_type *csrColIndC, value_type *csrValC)
{
    int err = 0;

    for (int i = 0; i < NUM_PLATFORMS; i++)
    {
        if (_spgemm_platform[i])
        {
            switch (i)
            {
            case BHSPARSE_CUDA:
                err = _bh_sparse_cuda->get_C(csrColIndC, csrValC);
                break;
            case BHSPARSE_OPENCL:
                //err = _bh_sparse_opencl->get_C(csrColIndC, csrValC);
                break;
            }
        }
    }
    if(err != BHSPARSE_SUCCESS) { cout << "get_C error = " << err << endl; return err; }

    return err;
}

#endif // BHSPARSE_H
