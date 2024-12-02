#include "AutomatonCUDA.h"

/* Kernels */
__global__ void collision_kernel(cudaTextureObject_t initial_grid_tex, uint16_t* outputs, int width, int height)
{
	// Calculate thread's 2D position in the grid
	// int x = blockIdx.x * blockDim.x + threadIdx.x;
	// int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Read data from the texture object
	// uint16_t cell_data = tex2D<uint16_t>(initial_grid_tex, x, y);

	 // Calculate global thread ID
	// int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	// Print thread information
	// if (threadIdx.x == 0 && blockIdx.x == 0) // Only first thread in the first block
		// printf("Thread ID: %d (Block: %d, Thread: %d)\n", threadIdx.x, blockIdx.x, threadIdx.x);
}

__global__ void streaming_kernel(cudaTextureObject_t outputs_tex, uint16_t* inputs, int width, int height)
{
	// Definition
}

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

	// Allocate memory on GPU
	allocate_memory();
}

// Destructor: Free device memory
AutomatonCUDA::~AutomatonCUDA()
{
	if (!cuda_available)
		return;
	
	// Free all allocated memory
	if (d_cells)
		cudaFree(d_cells);

	if (inputs)
		cudaFree(inputs);
}

// Copy initial state to GPU
void AutomatonCUDA::send(const uint16_t* h_cells)
{
	if (!cuda_available)
		return;

	// Copy state of the grid into GPU
	int alloc_size = width * height * sizeof(uint16_t);
	cudaMemcpy(d_cells, h_cells, alloc_size, cudaMemcpyHostToDevice);
}

// Perform GPU based update
void AutomatonCUDA::update()
{
	if (!cuda_available)
		return;

	// Define block and grid dimensions
	dim3 threads_per_block(16, 16);
	dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
		(height + threads_per_block.y - 1) / threads_per_block.y);
}

// Copy GPU's state to CPU
void AutomatonCUDA::retrieve(uint16_t* h_cells)
{
	if (!cuda_available)
		return;

	// Copy data from device to host
	int alloc_size = width * height * sizeof(uint16_t);
	cudaMemcpy(h_cells, d_cells, alloc_size, cudaMemcpyDeviceToHost);
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
	// Allocation for each cell
	int alloc_size = width * height * sizeof(uint16_t);
	cudaError_t return_code = cudaMalloc(&d_cells, alloc_size);
	print_malloc_failure(return_code, "d_cells", alloc_size);

	// Allocation for results in streaming kernel
	alloc_size = width * height * sizeof(uint16_t) * 5;
	cudaMalloc(&inputs, alloc_size);
	print_malloc_failure(return_code, "d_cells", alloc_size);
}

void AutomatonCUDA::print_malloc_failure(cudaError_t success_code, std::string name, int size)
{
	if (success_code != cudaSuccess)
		std::cerr << "CUDA malloc for " + name + " failed for size " << size << ": " << cudaGetErrorString(success_code) << "\n";
}
