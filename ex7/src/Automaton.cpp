#include "Automaton.h"

/* Constructor & Destructor */
Automaton::Automaton(int width, int height) : 
    width(width), height(height), cuda_helper(width, height), grid(width, height), grid_fallback(width, height){}

Automaton::~Automaton()
{}

/* Public Methods */
void Automaton::generate_random(double probability)
{
	int wall_position = this->width / 2; // Adjust proportion for wall position
	int gas_end = wall_position;        // End of gas region
	int wall_start = wall_position;     // Start of wall region
	int wall_end = wall_start + 1;      // End of wall region (1-cell wide)

	for (int y = 0; y < this->height; ++y)
	{
		for (int x = 0; x < this->width; ++x)
		{
			if (x < gas_end) // Gas-filled region
			{
				float rand_val = static_cast<float>(rand()) / RAND_MAX;

				if (rand_val < probability || probability == 1.f)
				{
					grid.set_cell_as_active(x, y);
				}
				else
				{
					grid.set_cell_as_inactive(x, y);
				}
			}
			else if (x >= wall_start && x < wall_end) // Wall region
			{
				grid.set_cell_as_wall(x, y);
			}
			else // Empty region
			{
				grid.set_cell_as_inactive(x, y);
			}

		}
	}

	// Save the copy of the grid
	grid_fallback.~Grid();	// Destruct
	new (&grid_fallback) Grid(grid);	// Reconstruct
}

void Automaton::reset()
{
	grid.~Grid();	// Destruct
	new (&grid) Grid(grid_fallback);	// Reconstruct
}

void Automaton::update(bool use_gpu)
{
	if (use_gpu)
		update_gpu();
	else
		update_cpu();
}

/* Private Methods */
void Automaton::update_cpu()
{
	// X-axis loop
	for (int x = 0; x < width; x++)
	{
		// Y-axis loop
		for (int y = 0; y < height; y++)
		{
			// Get this cell's id
			int cell_id = grid.get_id(x, y);

			// Skip if cell is a wall
			if (grid.is_wall[cell_id])
				continue;

			// Update functions of this cell
			for (int direction = 0; direction < grid.direction_num; direction++)
			{
				// 1. Collision
				// Equlibrium function
				grid.f_eq[direction][cell_id] = grid.weights[direction] * grid.concentration[cell_id];

				// Output function
				grid.f_out[direction][cell_id] = grid.f_in[direction][cell_id] + 1.0 / grid.tau * (grid.f_eq[direction][cell_id] - grid.f_in[direction][cell_id]);

				// 2. Streaming
				// Find the neighbour position
				int offset_x = grid.directions_x[direction];
				int offset_y = grid.directions_y[direction];

				int neighbour_x = x + offset_x;
				int neighbour_y = y + offset_y;

				// Check if the neighbour is out of bounds or a wall
				int neighbour_id = grid.get_id(neighbour_x, neighbour_y);
				if (neighbour_x < 0 || neighbour_x >= grid.width ||
					neighbour_y < 0 || neighbour_y >= grid.height ||
					grid.is_wall[neighbour_id])

				{	// Bounce Back
					int opposite_dir = grid.opposite_directions[direction];
					grid.f_in[opposite_dir][cell_id] = grid.f_out[direction][cell_id];
				}
				else // Within bounds
				{
					grid.f_in[direction][neighbour_id] = grid.f_out[direction][cell_id];
				}
			}
		}
	}

	// Calculate Concentration for each cell
	for (int i = 0; i < width * height; i++)
	{
		if (grid.is_wall[i])
		{
			grid.concentration[i] = 0.f;
			continue;
		}

		// Get the sum of all input directions
		double input_sum = 0.f;
		for (int dir = 0; dir < Grid::direction_num; dir++)
			input_sum += grid.f_in[dir][i];

		// Set the concetration of this cell
		grid.concentration[i] = input_sum; // / (double)Grid::direction_num;
	}
}

void Automaton::update_gpu()
{
	
}

/* Getters */
Grid* Automaton::get_grid()
{
	return &grid;
}