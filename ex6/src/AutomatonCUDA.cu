#include "AutomatonCUDA.h"

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

	// Set up texture memory
	this->res_desc = {};
	res_desc.resType = cudaResourceTypeArray;

	this->tex_desc = {};
	tex_desc.addressMode[0] = cudaAddressModeClamp; // Clamp outside boundary
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.filterMode = cudaFilterModePoint;     // No interpolation
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = 0;                 // Use normalized coordinates

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

	// Define block and grid dimensions
	dim3 threads_per_block(16, 16);
	dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
		(height + threads_per_block.y - 1) / threads_per_block.y);

	// Collision Phase: Create a texture for fast read-only memory with initial grid
	cudaTextureObject_t initial_grid_tex;
	res_desc.res.array.array = d_initial_grid;
	cudaCreateTextureObject(&initial_grid_tex, &res_desc, &tex_desc, nullptr);

	// Collision Phase: Launch Kernel
	collision_kernel <<< num_blocks, threads_per_block >>> (initial_grid_tex, d_outputs, width, height);

	// Collision Phase: Free texture from memory
	cudaDestroyTextureObject(initial_grid_tex);

	// Streaming Phase: Create a texture for fast read-only memory with outputs
	cudaTextureObject_t outputs_tex;
	res_desc.res.array.array = d_outputs_tex; // Bind outputs as the new texture
	cudaCreateTextureObject(&outputs_tex, &res_desc, &tex_desc, nullptr);

	// Streaming Phase: Launch kernel

	// Streaming Phase: Free texture from memory
	cudaDestroyTextureObject(outputs_tex);

	// Consolidation Phase: Combine the results sequentially
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

/* Kernels */
__global__ void collision_kernel(cudaTextureObject_t initial_grid_tex, uint16_t* outputs, int width, int height)
{
	// Calculate thread's 2D position in the grid
	// int x = blockIdx.x * blockDim.x + threadIdx.x;
	// int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Read data from the texture object
	// uint16_t cell_data = tex2D<uint16_t>(initial_grid_tex, x, y);
}

__global__ void streaming_kernel(cudaTextureObject_t outputs_tex, uint16_t* inputs, int width, int height)
{
	return __global__ void();
}
