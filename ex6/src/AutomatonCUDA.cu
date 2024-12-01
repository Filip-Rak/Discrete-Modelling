#include "AutomatonCUDA.h"

// Constructor: Set up device memory
AutomatonCUDA::AutomatonCUDA(int width, int height) : 
	width(width), height(height)
{
	// Check CUDA availability
	cuda_available = check_CUDA_availability();;

	// Set memory as NULL if CUDA not present
	if (!cuda_available)
	{
		d_cells = nullptr;
		return;
	}

	// Allocate memory with cuda
    size_t alloc_size = width * height * sizeof(uint16_t);
	cudaError_t return_code = cudaMalloc(&d_cells, alloc_size);

    if (return_code != cudaSuccess)
        std::cerr << "CUDA malloc failed for size " << alloc_size << ": " << cudaGetErrorString(return_code) << "\n";
}

// Destructor: Free device memory
AutomatonCUDA::~AutomatonCUDA()
{
	if (!cuda_available)
		return;
	
	cudaFree(d_cells);
}

// Copy initial state to GPU
void AutomatonCUDA::send(const uint16_t* h_cells)
{
	if (!cuda_available)
		return;

	cudaMemcpy(d_cells, h_cells, width * height * sizeof(uint16_t), cudaMemcpyHostToDevice);
}

// Perform GPU based update
void AutomatonCUDA::update()
{
	if (!cuda_available)
		return;

	// Call kernels
}

void AutomatonCUDA::retrieve(uint16_t* h_cells)
{
	if (!cuda_available)
		return;

	cudaMemcpy(h_cells, d_cells, width * height * sizeof(uint16_t), cudaMemcpyDeviceToHost);
}

bool AutomatonCUDA::check_CUDA_availability()
{
	int device_count = 0;
	cudaError_t err = cudaGetDeviceCount(&device_count);

	if (err != cudaSuccess || device_count == 0)
	{
		std::cerr << "No NVIDIA GPU detected or CUDA runtime is not available! GPU will not be used.\n";
		return false;
	}

	std::cout << "CUDA Available with: " << device_count << " device(s).\n";
	return true;
}
