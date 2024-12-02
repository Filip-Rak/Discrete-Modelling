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
		// d_cells = nullptr;
		return;
	}

	allocate_memory();
}

// Destructor: Free device memory
AutomatonCUDA::~AutomatonCUDA()
{
	if (!cuda_available)
		return;
	
	// Free all allocated memory
	if (d_initial_grid)
		cudaFreeArray(d_initial_grid);

	if (d_outputs_tex)
		cudaFreeArray(d_outputs_tex);

	if (d_outputs)
		cudaFree(d_outputs);

	if (d_inputs)
		cudaFree(d_inputs);

	if (results)
		cudaFreeHost(results);

}

// Copy initial state to GPU
void AutomatonCUDA::send(const uint16_t* h_cells)
{
	if (!cuda_available)
		return;

	// Find the size of the data to copy
	size_t size = width * height * sizeof(uint16_t);

	// Copy the host data to the CUDA array bound to the initial grid
	cudaError_t return_code = cudaMemcpyToArray(d_initial_grid, 0, 0, h_cells, size, cudaMemcpyHostToDevice);
}

// Perform GPU based update
void AutomatonCUDA::update()
{
	if (!cuda_available)
		return;

	// Call kernels
}

// Copy GPU's state to CPU
void AutomatonCUDA::retrieve(uint16_t* h_cells)
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

void AutomatonCUDA::allocate_memory()
{
	// Set texture memory as 16 bit
	channel_desc = cudaCreateChannelDesc<uint16_t>();

	// Allocate CUDA array for initial grid (read-only texture memory)
	size_t alloc_size = width * height * sizeof(uint16_t);
	cudaError_t return_code = cudaMallocArray(&d_initial_grid, &channel_desc, width, height);
	print_malloc_failure(return_code, "initial grid (read-only)", alloc_size);

	// Allocate device memory for outputs
	return_code = cudaMalloc(&d_outputs, alloc_size);
	print_malloc_failure(return_code, "outputs", alloc_size);

	// Allocate CUDA array for outputs as read-only texture
	return_code = cudaMallocArray(&d_outputs_tex, &channel_desc, width, height);
	print_malloc_failure(return_code, "outputs (read-only)", alloc_size);

	// Allocate device memory for inputs
	alloc_size = width * height * 5 * sizeof(uint16_t);
	return_code = cudaMalloc(&d_inputs, alloc_size);
	print_malloc_failure(return_code, "inputs", alloc_size);
	
	// Allocate pinned host memory for results
	return_code = cudaHostAlloc(&results, alloc_size, cudaHostAllocMapped);
	print_malloc_failure(return_code, "results", alloc_size);
}

void AutomatonCUDA::print_malloc_failure(cudaError_t success_code, std::string name, int size)
{
	if (success_code != cudaSuccess)
		std::cerr << "CUDA malloc for " + name + " failed for size " << size << ": " << cudaGetErrorString(success_code) << "\n";
}
