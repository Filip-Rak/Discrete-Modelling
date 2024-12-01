#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

class AutomatonCUDA
{
private:
	/* Attributes */
	uint16_t* d_cells;	// Device memory for cells
	int width, height;	// Dimensions of the grid

	// Flags
	bool cuda_available;

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
};