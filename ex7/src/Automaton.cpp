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
void Automaton::generate_random()
{
}
