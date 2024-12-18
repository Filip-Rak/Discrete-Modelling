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
		f_eq[i] = new double[total_cells]();
		f_out[i] = new double[total_cells]();
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

		f_eq[i] = new double[total_cells];
		std::memcpy(f_eq[i], other.f_eq[i], total_cells * sizeof(double));

		f_out[i] = new double[total_cells];
		std::memcpy(f_out[i], other.f_out[i], total_cells * sizeof(double));
	}
}

Grid::~Grid()
{
	delete[] concentration;
	delete[] is_wall;

	for (int i = 0; i < direction_num; ++i)
	{
		delete[] f_in[i];
		delete[] f_eq[i];
		delete[] f_out[i];
	}
}

void Grid::set_cell_as_active(int x, int y)
{
	// Get the cell's index
	int cell_id = y * width + x;

	// Set cell's properties
	is_wall[cell_id] = false;

	double input_sum = 0.f;
	for (int j = 0; j < direction_num; j++)
	{
		double addition = 1.f;	// Work on this var to change the state of activation
		f_in[j][cell_id] = addition;
		input_sum += addition;

		// Zero other arrays
		f_eq[j][cell_id];
		f_out[j][cell_id];
	}

	concentration[cell_id] = input_sum / (double)direction_num;
}

void Grid::set_cell_as_inactive(int x, int y)
{
	// Get the cell's index
	int cell_id = y * width + x;

	// Set cell's properties
	concentration[cell_id] = 0.f;
	is_wall[cell_id] = false;

	for (int j = 0; j < direction_num; j++)
	{
		f_in[j][cell_id] = 0.f;
		f_eq[j][cell_id] = 0.f;
		f_out[j][cell_id] = 0.f;
	}
}

void Grid::set_cell_as_wall(int x, int y)
{
	// Get the cell's index
	int cell_id = y * width + x;

	// Set cell's properties
	concentration[cell_id] = 0.f;
	is_wall[cell_id] = true;

	for (int j = 0; j < direction_num; j++)
	{
		f_in[j][cell_id] = 0.f;
		f_eq[j][cell_id] = 0.f;
		f_out[j][cell_id] = 0.f;
	}
}

double Grid::get_cell_concetration(int x, int y)
{
	// Get the cell's index
	int cell_id = y * width + x;

	// Return value
	return concentration[cell_id];
}

bool Grid::get_cell_is_wall(int x, int y)
{
	// Get the cell's index
	int cell_id = y * width + x;

	// Return value
	return is_wall[cell_id];
}
