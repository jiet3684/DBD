#include <cuda_runtime.h>
#include "Executor.h"
#include "Multiply.h"
#include "DataLoader.h"
#include <iomanip>
#include "Config.h"
#include "Compare.h"
//#include <cusparse/include/cuSparseMultiply.h>
#include "Timings.h"
#include "spECKConfig.h"
#include <iostream>
#include <sys/time.h>

#define BLOCKSIZE 256

void fileWrite(int *ptr, int *idx, float *val, int nr, int ne) {
	FILE *fp = fopen("output", "wb");
	/*int nr = block.rows;
	int ne = block.nnz;
	int *ptr = block.row_offsets.get();
	int *idx = block.col_ids.get();
	float *val = block.data.get();*/

	fwrite(ptr, sizeof(int), nr + 1, fp);
	fwrite(idx, sizeof(int), ne, fp);
	fwrite(val, sizeof(float), ne, fp);

	fclose(fp);
}


template <typename ValueType>
int Executor<ValueType>::run()
{
	iterationsWarmup = Config::getInt(Config::IterationsWarmUp, 5);
	//iterationsExecution = Config::getInt(Config::IterationsExecution, 10);
	iterationsExecution = 3;
	DataLoader<ValueType> data(runConfig.filePath);
	auto& matrices = data.matrices;
	std::cout << "Matrix: " << matrices.cpuA.rows << "x" << matrices.cpuA.cols << ": " << matrices.cpuA.nnz << " nonzeros\n";

	dCSR<ValueType> dCsrHiRes, dCsrReference;
	Timings timings, warmupTimings, benchTimings;
	//bool measureAll = Config::getBool(Config::TrackIndividualTimes, false);
	bool measureAll = Config::getBool(Config::TrackIndividualTimes, false);
	bool measureCompleteTimes = Config::getBool(Config::TrackCompleteTimes, true);
	auto config = spECK::spECKConfig::initialize(0);

	bool compareData = false;

	/*if(Config::getBool(Config::CompareResult))
	{
		unsigned cuSubdiv_nnz = 0;
		cuSPARSE::CuSparseTest<ValueType> cusparse;
		cusparse.Multiply(matrices.gpuA, matrices.gpuB, dCsrReference, cuSubdiv_nnz);

		if(!compareData)
		{
			cudaFree(dCsrReference.data);
			dCsrReference.data = nullptr;
		}
	}*/

	// Warmup iterations for multiplication
	for (int i = 0; i < iterationsWarmup; ++i)
	{
		timings = Timings();
		timings.measureAll = measureAll;
		timings.measureCompleteTime = measureCompleteTimes;
		spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
		warmupTimings += timings;

		if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr && Config::getBool(Config::CompareResult))
		{
			if (!spECK::Compare(dCsrReference, dCsrHiRes, false))
				printf("Error: Matrix incorrect\n");
		}
	}
	//size_t nr = dCsrHiRes.rows;
	//size_t nc = dCsrHiRes.cols;
	//size_t ne = dCsrHiRes.nnz;

	size_t nr = matrices.cpuA.rows;
	size_t nc = matrices.cpuA.cols;
	size_t ne = matrices.cpuA.nnz;



	struct timeval st, ed;
	float t_kernel = 0, t_d2h = 0, t_write = 0;
	int num_Blocks = ((nr - 1) / BLOCKSIZE) + 1; 

	int *block_Ptr = (int*)malloc(sizeof(int) * (BLOCKSIZE + 1));

	for (int i = 0; i < iterationsExecution; ++i)
	{
		gettimeofday(&st, NULL);
		timings = Timings();
		timings.measureAll = measureAll;
		timings.measureCompleteTime = measureCompleteTimes;

		spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
		benchTimings += timings;

		if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr && Config::getBool(Config::CompareResult))
		{
			if (!spECK::Compare(dCsrReference, dCsrHiRes, false))
				printf("Error: Matrix incorrect\n");
		}
		gettimeofday(&ed, NULL);
		t_kernel += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
		printf("%f\t", (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec));
		
		int *ptr = (int*)malloc(sizeof(int) * (dCsrHiRes.rows + 1));
		int *idx = (int*)malloc(sizeof(int) * dCsrHiRes.nnz);
		float *val = (float*)malloc(sizeof(float) * dCsrHiRes.nnz);
		//printf("%d %d %d\n", dCsrHiRes.rows, dCsrHiRes.cols, dCsrHiRes.nnz);
		spECK::TransferResult(ptr, idx, val, dCsrHiRes, dCsrHiRes.rows, dCsrHiRes.nnz);
		gettimeofday(&ed, NULL);
		t_d2h += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
		printf("%f\t", (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec));

		fileWrite(ptr, idx, val, dCsrHiRes.rows, dCsrHiRes.nnz);
		gettimeofday(&ed, NULL);
		t_write += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
		printf("%f\t%lu\n", (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec), dCsrHiRes.nnz);
		free(ptr);
		free(idx);
		free(val);
	}

	// Multiplication
	/*for (int i = 0; i < iterationsExecution; ++i)
	{
		printf("Iter %d\t", i);
		unsigned long nnzC = 0;
		for (int j = 0; j < num_Blocks; ++j) {
				
			timings = Timings();
			timings.measureAll = measureAll;
			timings.measureCompleteTime = measureCompleteTimes;

			int start_Row = j * BLOCKSIZE;
			int end_Row = start_Row + BLOCKSIZE;
			if (j == num_Blocks - 1) end_Row = nr;
			int nr_in_this_block = end_Row - start_Row;

			matrices.gpuA.rows = nr_in_this_block;
			matrices.gpuA.nnz = matrices.cpuA.row_offsets[end_Row] - matrices.cpuA.row_offsets[start_Row];
			for (int k = 0; k < nr_in_this_block + 1; ++k) block_Ptr[k] = matrices.cpuA.row_offsets[start_Row + k] - matrices.cpuA.row_offsets[start_Row];
			cudaMemcpy(matrices.gpuA.row_offsets, &block_Ptr[0], sizeof(int) * (nr_in_this_block + 1), cudaMemcpyDeviceToHost);
			cudaMemcpy(matrices.gpuA.col_ids, &matrices.cpuA.col_ids[matrices.cpuA.row_offsets[start_Row]], sizeof(int) * matrices.gpuA.nnz, cudaMemcpyDeviceToHost);
			cudaMemcpy(matrices.gpuA.data, &matrices.cpuA.data[matrices.cpuA.row_offsets[start_Row]], sizeof(float) * matrices.gpuA.nnz, cudaMemcpyDeviceToHost);

			gettimeofday(&st, NULL);
			spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
			benchTimings += timings;


			gettimeofday(&ed, NULL);
			t_kernel += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);
			//printf("%f\t", (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec));
			
			//printf("%d %d %d\n", dCsrHiRes.rows, dCsrHiRes.cols, dCsrHiRes.nnz);
			//CSR<float> temp;
			//convert<float>(temp, dCsrHiRes, 0);
			nnzC += dCsrHiRes.nnz;
			int *ptr = (int*)malloc(sizeof(int) * (dCsrHiRes.rows + 1));
			int *idx = (int*)malloc(sizeof(int) * dCsrHiRes.nnz);
			float *val = (float*)malloc(sizeof(float) * dCsrHiRes.nnz);
			spECK::TransferResult(ptr, idx, val, dCsrHiRes, dCsrHiRes.rows, dCsrHiRes.nnz);
			gettimeofday(&ed, NULL);
			t_d2h += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);

			fileWrite(ptr, idx, val, dCsrHiRes.rows, dCsrHiRes.nnz);
			gettimeofday(&ed, NULL);
			t_write += (float)(ed.tv_sec - st.tv_sec) + 0.000001 * (float)(ed.tv_usec - st.tv_usec);

			free(ptr);
			free(idx);
			free(val);

		}
		printf("%lu\n", nnzC);

	}*/
	puts("\nAverage:" );
	printf("Kernel %f\n", t_kernel / iterationsExecution);
	printf("D2H %f\n", t_d2h / iterationsExecution);
	printf("Write %f\n", t_write / iterationsExecution);
	
	benchTimings /= iterationsExecution;

	free(block_Ptr);

	//std::cout << std::setw(20) << "var-SpGEMM -> NNZ: " << dCsrHiRes.nnz << std::endl;
	//std::cout << std::setw(20) << "var-SpGEMM SpGEMM: " << benchTimings.complete * 0.001 << " s" << std::endl;



	return 0;
}

template int Executor<float>::run();
//template int Executor<double>::run();
