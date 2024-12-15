#include "AutomatonCUDA.h"
#include "Automaton.h"

// Constructor: Set up device memory
AutomatonCUDA::AutomatonCUDA(int width, int height) : 
	width(width), height(height)
{
	// Check CUDA availability
	cuda_available = check_CUDA_availability();;

	// Stop execution if CUDA not available
	if (!cuda_available)
	{
		return;
	}

	// Allocate memory
}

// Destructor: Free device memory
AutomatonCUDA::~AutomatonCUDA()
{
	if (!cuda_available)
		return;
}

/* Private Methods */

// Check if CUDA is available on this device
bool AutomatonCUDA::check_CUDA_availability()
{
	int device_count = 0;
	cudaError_t err = cudaGetDeviceCount(&device_count);

	if (err != cudaSuccess || device_count == 0)
	{
		return false;
	}

	std::cout << "CUDA Available with: " << device_count << " device(s).\n";
	return true;
}