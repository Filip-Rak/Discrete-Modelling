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

	int width, height;	// Dimensions of the grid

	// Flags
	bool cuda_available;	// Blocks all cuda functions if true

public:
	/* Constructor */
	AutomatonCUDA(int width, int height);
	~AutomatonCUDA();

	/* Public Methods */
	void send();	// Copy host's state to GPU
	void update();						// Perform GPU based update
	void retrieve();	// Copy updated state back to CPU

private:
	/* Private Methods */
	bool check_CUDA_availability();
	void allocate_memory();
	void print_malloc_failure();
	void combine_local_neighbours(uint16_t* h_cells, uint16_t* h_inputs);
};