#include "Grid.h"

/* Constructors */
Grid::Grid(int w, int h)
	: width(w), height(h), concentration(nullptr), is_wall(nullptr)
{
	int total_cells = w * h;

	// Allocate and initialize with zeros
	concentration = new double[total_cells]();
	is_wall = new bool[total_cells]();

	for (int i = 0; i < direction_num; ++i) 
	{
		f_in[i] = new double[total_cells]();
	}
}

Grid::Grid(const Grid& other)
	: width(other.width), height(other.height), concentration(nullptr), is_wall(nullptr)
{
	int total_cells = width * height;

	concentration = new double[total_cells];
	std::memcpy(concentration, other.concentration, total_cells * sizeof(double));

	is_wall = new bool[total_cells];
	std::memcpy(is_wall, other.is_wall, total_cells * sizeof(bool));

	for (int i = 0; i < direction_num; ++i)
	{
		f_in[i] = new double[total_cells];
		std::memcpy(f_in[i], other.f_in[i], total_cells * sizeof(double));
	}
}

Grid::~Grid()
{
	delete[] concentration;
	delete[] is_wall;

	for (int i = 0; i < direction_num; ++i)
	{
		delete[] f_in[i];
	}
}

/* Static Methods */
int Grid::get_id(int x_pos, int y_pos)
{
	// Return the cell's index in the array
	return y_pos * width + x_pos;
}

/* Setters */
void Grid::set_cell_as_active(int x, int y)
{
	// Get the cell's index
	int cell_id = get_id(x, y);

	// Call the setter
	set_cell_as_active(cell_id);
}

void Grid::set_cell_as_active(int cell_id)
{
	// Set cell's properties
	is_wall[cell_id] = false;

	double input_sum = 0.f;
	for (int j = 0; j < direction_num; j++)
	{
		double addition = 1.f;	// Work on this var to change the state of activation
		f_in[j][cell_id] = addition;
		input_sum += addition;
	}

	concentration[cell_id] = input_sum; // / (double)direction_num;
}

void Grid::set_cell_as_inactive(int x, int y)
{
	// Get the cell's index
	int cell_id = get_id(x, y);

	// Call the setter
	set_cell_as_inactive(cell_id);
}

void Grid::set_cell_as_inactive(int cell_id)
{
	// Set cell's properties
	concentration[cell_id] = 0.f;
	is_wall[cell_id] = false;

	for (int j = 0; j < direction_num; j++)
	{
		f_in[j][cell_id] = 0.f;
	}
}

void Grid::set_cell_as_wall(int x, int y)
{
	// Get the cell's index
	int cell_id = get_id(x, y);

	// Call the setter
	set_cell_as_wall(cell_id);
}

void Grid::set_cell_as_wall(int cell_id)
{
	// Set cell's properties
	concentration[cell_id] = 0.f;
	is_wall[cell_id] = true;

	for (int j = 0; j < direction_num; j++)
	{
		f_in[j][cell_id] = 0.f;
	}
}

/* Getters */
double Grid::get_cell_concetration(int x, int y)
{
	// Get the cell's index
	int cell_id = get_id(x, y);

	// Call the getter
	return get_cell_concetration(cell_id);
}

double Grid::get_cell_concetration(int cell_id)
{
	return concentration[cell_id];
}

bool Grid::get_cell_is_wall(int x, int y)
{
	// Get the cell's index
	int cell_id = get_id(x, y);

	// Return value
	return get_cell_is_wall(cell_id);
}

bool Grid::get_cell_is_wall(int cell_id)
{
	return is_wall[cell_id];
}
