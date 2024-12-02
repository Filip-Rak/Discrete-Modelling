#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <cstdint>
#include <iostream>

class AutomatonCUDA
{
private:
	/* Attributes */

	// Arrays
	cudaChannelFormatDesc channel_desc;	// Channel descriptor for textures
	cudaArray* d_initial_grid;	// Texture memory with state given to the automaton (read-only)
	cudaArray* d_outputs_tex;	// Texture memory for outputs during streaming (read-only)
	uint16_t* d_outputs;		// Writeable memory for outputs (GPU)
	uint16_t* d_inputs;			// Writeable memory for inputs (GPU, 2D Array)
	uint16_t* results;			// Memory for results, used by CPU

	// Texture memory setting
	cudaResourceDesc res_desc;
	cudaTextureDesc tex_desc;

	int width, height;	// Dimensions of the grid

	// Flags
	bool cuda_available;	// Blocks all cuda functions if true

public:
	/* Constructor */
	AutomatonCUDA(int width, int height);
	~AutomatonCUDA();

	/* Public Methods */
	void send(const uint16_t* h_cells);	// Copy host's state to GPU
	void update();						// Perform GPU based update
	void retrieve(uint16_t* h_cells);	// Copy updated state back to CPU

private:
	/* Private Methods */
	bool check_CUDA_availability();
	void allocate_memory();
	void print_malloc_failure(cudaError_t success_code, std::string name, int size);
};

/* Kernels | Outside the class */
__global__ void collision_kernel(cudaTextureObject_t tex_obj, uint16_t* outputs, int width, int height);
__global__ void streaming_kernel(cudaTextureObject_t tex_obj, uint16_t* inputs, int width, int height);