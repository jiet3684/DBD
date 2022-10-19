#include "DataLoader.h"

#include <iostream>
#include "COO.h"
#include "cusparse/include/cuSparseMultiply.h"

template<typename T>
std::string typeExtension();
template<>
std::string typeExtension<float>()
{
	return std::string("");
}
template<>
std::string typeExtension<double>()
{
	return std::string("d_");
}

template class DataLoader<float>;
template class DataLoader<double>;

template <typename ValueType>
DataLoader<ValueType>::DataLoader(std::string path) : matrices()
{
	std::string csrPath = path + typeExtension<ValueType>() + ".hicsr";

	try
	{
		std::cout << "trying to load csr file \"" << csrPath << "\"\n";
		matrices.cpuA = loadCSR<ValueType>(csrPath.c_str());
		std::cout << "successfully loaded: \"" << csrPath << "\"\n";
	}
	catch (std::exception& ex)
	{
		std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
		try
		{
			std::cout << "trying to load mtx file \"" << path << "\"\n";
			COO<ValueType> cooMat = loadMTX<ValueType>(path.c_str());
			convert(matrices.cpuA, cooMat);
			std::cout << "successfully loaded and converted: \"" << csrPath << "\"\n";
		}
		catch (std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
			std::cout << "could not load mtx file: \"" << path << "\"\n";
			throw "could not load mtx file";
		}
		try
		{
			std::cout << "write csr file for future use\n";
			storeCSR(matrices.cpuA, csrPath.c_str());
		}
		catch (std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
		}
	}
	
	convert(matrices.gpuA, matrices.cpuA, 0);
	cuSPARSE::CuSparseTest<ValueType> cuSparse;
	
	convert(matrices.gpuB, matrices.cpuA, 0);
	convert(matrices.cpuB, matrices.cpuA, 0);
}