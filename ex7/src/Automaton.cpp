#include "Automaton.h"

/* Constructor & Destructor */
Automaton::Automaton(int width, int height) : 
    cuda_helper(width, height)
{
	/* Create the grid */

	// Allocate memory
	int cell_number = width * height;

	grid.concentration = new double[cell_number];
	grid.is_wall = new bool[cell_number];

	for (int dir = 0; dir < direction_num; dir++) 
	{
		grid.f_in[dir] = new double[cell_number];
		grid.f_eq[dir] = new double[cell_number];
		grid.f_out[dir] = new double[cell_number];
	}

	// Initialize memory
	grid.width = width;
	grid.height = height;

	for (int i = 0; i < cell_number; i++)
	{
		grid.concentration[i] = 0.f;
		grid.is_wall[i] = false;

		for (int j = 0; j < direction_num; j++)
		{
			grid.f_in[j][i] = 0.f;
			grid.f_eq[j][i] = 0.f;
			grid.f_out[j][i] = 0.f;
		}
	}
}

Automaton::~Automaton()
{
	// Free grid dynamic data
	delete[] grid.concentration;
	delete[] grid.is_wall;

	for (int dir = 0; dir < direction_num; dir++)
	{
		delete[] grid.f_in[dir];
		delete[] grid.f_eq[dir];
		delete[] grid.f_out[dir];
	}
}

/* Public Methods */
void Automaton::generate_random(double probability)
{
	int wall_position = this->grid.width / 3; // Adjust proportion for wall position
	int gas_end = wall_position;        // End of gas region
	int wall_start = wall_position;     // Start of wall region
	int wall_end = wall_start + 1;      // End of wall region (1-cell wide)

	for (int y = 0; y < this->grid.height; ++y)
	{
		for (int x = 0; x < this->grid.width; ++x)
		{
			if (x < gas_end) // Gas-filled region
			{
				float rand_val = static_cast<float>(rand()) / RAND_MAX;

				if (rand_val < probability || probability == 1.f)
				{
					set_cell_as_active(x, y);
				}
				else
				{
					set_cell_as_inactive(x, y);
				}
			}
			else if (x >= wall_start && x < wall_end) // Wall region
			{
				set_cell_as_wall(x, y);
			}
			else // Empty region
			{
				set_cell_as_inactive(x, y);
			}

		}
	}

	// Save the copy of the grid
	// (...)
	std::cout << "ERROR: Automaton::generate_random() -> Function does not handle grid copy for fallback\n";
}

void Automaton::reset()
{
	std::cout << "ERROR: Automaton::reset() -> No function definition\n";
}

/* Getters */
Automaton::Grid* Automaton::get_grid()
{
	return &grid;
}

/* Setters */
void Automaton::set_cell_as_active(int x, int y)
{
	// Get the cell's index
	int cell_id = y * grid.width + x;

	// Set cell's properties
	grid.is_wall[cell_id] = false;

	double input_sum = 0.f;
	for (int j = 0; j < direction_num; j++)
	{
		double addition = 1.f;	// Work on this var to change the state of activation
		grid.f_in[j][cell_id] = addition;
		input_sum += addition;

		// Zero other arrays
		grid.f_eq[j][cell_id];
		grid.f_out[j][cell_id];
	}

	grid.concentration[cell_id] = input_sum / (double)direction_num;
}

void Automaton::set_cell_as_inactive(int x, int y)
{
	// Get the cell's index
	int cell_id = y * grid.width + x;

	// Set cell's properties
	grid.concentration[cell_id] = 0.f;
	grid.is_wall[cell_id] = false;

	for (int j = 0; j < direction_num; j++)
	{
		grid.f_in[j][cell_id] = 0.f;
		grid.f_eq[j][cell_id] = 0.f;
		grid.f_out[j][cell_id] = 0.f;
	}
}

void Automaton::set_cell_as_wall(int x, int y)
{
	// Get the cell's index
	int cell_id = y * grid.width + x;

	// Set cell's properties
	grid.concentration[cell_id] = 0.f;
	grid.is_wall[cell_id] = true;
	
	for (int j = 0; j < direction_num; j++)
	{
		grid.f_in[j][cell_id] = 0.f;
		grid.f_eq[j][cell_id] = 0.f;
		grid.f_out[j][cell_id] = 0.f;
	}
}