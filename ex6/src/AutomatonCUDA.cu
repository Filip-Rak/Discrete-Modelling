#include "AutomatonCUDA.h"
#include "Automaton.h"

// Bit format //
// 15 14 | 13 12 11 10 9 8 | 7 6 5 4 | 3 2 1 0
// State |      Empty	   | Outputs | Inputs
// State //
// WALL = 0b00 << 14,
// EMPTY = 0b01 << 14,
// GAS = 0b10 << 14
// Direction
// UP = 0,	  // Bit 0
// RIGHT = 1, // Bit 1
// LEFT = 2,  // Bit 2
// DOWN = 3	  // Bit 3

// inputs array format //
// [0] = cell's self input
// [1] = up neighbour's input
// [2] = down neighbour's input
// [3] = left neighbour's input
// [4] = right neighbour's input

/* Kernels */
__global__ void collision_kernel(uint16_t* d_cells, int width, int height)
{
	// Calculate thread's 2D position in the grid
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Calculate the 1D index for the cell
	int cell_id = y * width + x;

	// Load the cell data
	uint16_t cell = d_cells[cell_id];

	// Skip walls
	if (Automaton::get_state(cell) == Automaton::WALL)
		return;

	// Extract input directions
	uint8_t input = Automaton::get_input(cell);

	// Perform collision resolution using masks
	if ((input & UP_DOWN_MASK) == UP_DOWN_MASK && (input & ~UP_DOWN_MASK) == 0)
	{
		// Flip UP and DOWN to LEFT and RIGHT
		input = LEFT_RIGHT_MASK;
	}
	else if ((input & LEFT_RIGHT_MASK) == LEFT_RIGHT_MASK && (input & ~LEFT_RIGHT_MASK) == 0)
	{
		// Flip LEFT and RIGHT to UP and DOWN
		input = UP_DOWN_MASK;
	}

	// Update the cell with new output directions
	cell = Automaton::set_output(cell, input);

	// Write the updated cell back to global memory
	d_cells[cell_id] = cell;
}

__global__ void streaming_kernel(uint16_t* d_cells, uint16_t* inputs, int width, int height)
{
	// Calculate thread's 2D position in the grid
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Boundary check
	if (x >= width || y >= height)
		return;

	// Calculate the 1D index for the cell
	int cell_id = y * width + x;

	// Initialize local inputs for this cell to 0
	for (int i = 0; i < 5; i++)
		inputs[cell_id * 5 + i] = 0;

	// Load the cell data
	uint16_t cell = d_cells[cell_id];

	// Skip walls
	if (Automaton::get_state(cell) == Automaton::WALL)
		return;

	// Extract the output directions
	uint8_t output = Automaton::get_output(cell);

	// Propagate particles to neighbors
	if (output & (1 << Automaton::UP))
	{
		bool is_within_bounds = (y > 0);

		if (is_within_bounds)
		{
			int neighbor_id = (y - 1) * width + x;
			inputs[neighbor_id * 5 + 0] |= (1 << Automaton::UP);
		}
		else
		{
			// Save as local of this cell in opposite direction
			inputs[cell_id * 5 + 0] |= (1 << Automaton::DOWN);
		}
	}

	if (output & (1 << Automaton::DOWN))
	{
		bool is_within_bounds = (y < height - 1);

		if (is_within_bounds)
		{
			int neighbor_id = (y + 1) * width + x;
			inputs[neighbor_id * 5 + 0] |= (1 << Automaton::DOWN);
		}
		else
		{
			// Save as local of this cell in opposite direction
			inputs[cell_id * 5 + 0] |= (1 << Automaton::UP);
		}
	}

	if (output & (1 << Automaton::LEFT))
	{
		bool is_within_bounds = (x > 0);

		if (is_within_bounds)
		{
			int neighbor_id = y * width + (x - 1);
			inputs[neighbor_id * 5 + 0] |= (1 << Automaton::LEFT);
		}
		else
		{
			// Save as local of this cell in opposite direction
			inputs[cell_id * 5 + 0] |= (1 << Automaton::RIGHT);
		}
	}

	if (output & (1 << Automaton::RIGHT))
	{
		bool is_within_bounds = (x < width - 1);

		if (is_within_bounds)
		{
			int neighbor_id = y * width + (x + 1);
			inputs[neighbor_id * 5 + 0] |= (1 << Automaton::RIGHT);
		}
		else
		{
			// Save as local of this cell in opposite direction
			inputs[cell_id * 5 + 0] |= (1 << Automaton::LEFT);
		}
	}
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

	if (d_inputs)
		cudaFree(d_inputs);	
	
	if (h_inputs)
		delete[] h_inputs;
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

	// Collisions: Launch kernel
	collision_kernel <<< num_blocks, threads_per_block >>> (d_cells, width, height);
	cudaDeviceSynchronize();

	streaming_kernel <<< num_blocks, threads_per_block >>> (d_cells, d_inputs, width, height);
	cudaDeviceSynchronize();

	// Copy kernel's results to CPU's results
	int copy_size = width * height * 5 * sizeof(uint16_t);
	cudaMemcpy(h_inputs, d_inputs, copy_size, cudaMemcpyDeviceToHost);
}

// Copy GPU's state to CPU
void AutomatonCUDA::retrieve(uint16_t* h_cells)
{
	if (!cuda_available)
		return;

	// Copy the latest cell data from the device
	int alloc_size = width * height * sizeof(uint16_t);
	cudaMemcpy(h_cells, d_cells, alloc_size, cudaMemcpyDeviceToHost);

	// Combine local neighbor inputs and update h_cells
	combine_local_neighbours(h_cells, h_inputs);
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
	return_code = cudaMalloc(&d_inputs, alloc_size);
	print_malloc_failure(return_code, "d_inputs", alloc_size);

	// Allocation for CPU copy of streaming kernel's result
	alloc_size = width * height * sizeof(uint16_t) * 5;
	h_inputs = new uint16_t[width * height * 5];
}

void AutomatonCUDA::print_malloc_failure(cudaError_t success_code, std::string name, int size)
{
	if (success_code != cudaSuccess)
		std::cerr << "CUDA malloc for " + name + " failed for size " << size << ": " << cudaGetErrorString(success_code) << "\n";
}

void AutomatonCUDA::combine_local_neighbours(uint16_t* h_cells, uint16_t* h_inputs)
{
	// First, clear the input bits in h_cells
	for (int i = 0; i < width * height; ++i)
	{
		h_cells[i] = Automaton::set_input(h_cells[i], 0);
	}

	// Iterate over each cell in the grid
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			int cell_id = y * width + x;

			// Skip walls
			if (Automaton::get_state(h_cells[cell_id]) == Automaton::WALL)
				continue;

			// Apply the cell's own input to itself
			uint16_t self_input_cell = h_inputs[cell_id * 5 + 0];
			uint8_t self_input_bits = Automaton::get_input(self_input_cell);

			h_cells[cell_id] = Automaton::set_input(h_cells[cell_id], self_input_bits);

			// Distribute inputs to neighbor cells
			// Directions: 1 - UP, 2 - DOWN, 3 - LEFT, 4 - RIGHT

			// UP neighbor
			if (y > 0)
			{
				int up_id = (y - 1) * width + x;
				if (Automaton::get_state(h_cells[up_id]) != Automaton::WALL)
				{
					uint16_t up_input_cell = h_inputs[cell_id * 5 + 1];
					uint8_t up_input_bits = Automaton::get_input(up_input_cell);
					h_cells[up_id] = Automaton::set_input(h_cells[up_id],
						Automaton::get_input(h_cells[up_id]) | up_input_bits);
				}
			}
			// DOWN neighbor
			if (y < height - 1)
			{
				int down_id = (y + 1) * width + x;
				if (Automaton::get_state(h_cells[down_id]) != Automaton::WALL)
				{
					uint16_t down_input_cell = h_inputs[cell_id * 5 + 2];
					uint8_t down_input_bits = Automaton::get_input(down_input_cell);
					h_cells[down_id] = Automaton::set_input(h_cells[down_id],
						Automaton::get_input(h_cells[down_id]) | down_input_bits);
				}
			}
			// LEFT neighbor
			if (x > 0)
			{
				int left_id = y * width + (x - 1);
				if (Automaton::get_state(h_cells[left_id]) != Automaton::WALL)
				{
					uint16_t left_input_cell = h_inputs[cell_id * 5 + 3];
					uint8_t left_input_bits = Automaton::get_input(left_input_cell);
					h_cells[left_id] = Automaton::set_input(h_cells[left_id],
						Automaton::get_input(h_cells[left_id]) | left_input_bits);
				}
			}
			// RIGHT neighbor
			if (x < width - 1)
			{
				int right_id = y * width + (x + 1);
				if (Automaton::get_state(h_cells[right_id]) != Automaton::WALL)
				{
					uint16_t right_input_cell = h_inputs[cell_id * 5 + 4];
					uint8_t right_input_bits = Automaton::get_input(right_input_cell);
					h_cells[right_id] = Automaton::set_input(h_cells[right_id],
						Automaton::get_input(h_cells[right_id]) | right_input_bits);
				}
			}
		}
	}

	// Update the state of each cell based on new inputs
	for (int i = 0; i < width * height; ++i)
	{
		uint16_t cell = h_cells[i];

		// Skip walls
		if (Automaton::get_state(cell) == Automaton::WALL)
			continue;

		uint8_t input = Automaton::get_input(cell);
		uint8_t output = Automaton::get_output(cell);

		if (input != 0 || output != 0)
		{
			cell = Automaton::set_state(cell, Automaton::GAS);
		}
		else
		{
			cell = Automaton::set_state(cell, Automaton::EMPTY);
		}

		// Clear the output bits for the next iteration
		cell = Automaton::set_output(cell, 0);

		// Write back the updated cell
		h_cells[i] = cell;
	}
}

