#include "Automaton.h"

/* Constructor & Destructor */
Automaton::Automaton(int width, int height) : 
    width(width), height(height), cuda_helper(width, height), grid(width, height), grid_fallback(width, height){}

Automaton::~Automaton()
{}

/* Public Methods */
void Automaton::generate_random(double probability)
{
	int wall_position = this->width / 3; // Adjust proportion for wall position
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
	
}

void Automaton::update_gpu()
{
	
}

/* Getters */
Grid* Automaton::get_grid()
{
	return &grid;
}