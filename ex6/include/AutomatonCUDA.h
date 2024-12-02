#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <cstdint>
#include <iostream>

/* Kernels | Outside the class */
__global__ void collision_kernel(uint16_t* d_cells, int width, int height);

__global__ void streaming_kernel(uint16_t* d_cells, uint16_t* inputs, int width, int height);

/* Kernel Definitions */
__device__ constexpr uint8_t UP_DOWN_MASK = (1 << 0) | (1 << 3); // UP and DOWN
__device__ constexpr uint8_t LEFT_RIGHT_MASK = (1 << 1) | (1 << 2); // LEFT and RIGHT

class AutomatonCUDA
{
private:
	/* Attributes */

	// Arrays
	uint16_t* d_cells;
	uint16_t* d_inputs;
	uint16_t* h_inputs;

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
	void combine_local_neighbours(uint16_t* h_cells, uint16_t* h_inputs);
};