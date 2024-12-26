#include "Grid.h"

/* Constructors */
Grid::Grid(int w, int h)
	: width(w), height(h), density(nullptr), is_wall(nullptr)
{
	int total_cells = w * h;

	// Allocate and initialize with zeros
	density = new double[total_cells]();
	velocity_x = new double[total_cells]();
	velocity_y = new double[total_cells]();
	is_wall = new bool[total_cells]();

	for (int i = 0; i < direction_num; ++i) 
	{
		f_in[i] = new double[total_cells]();
		f_buffer[i] = new double[total_cells]();
	}
}

Grid::Grid(const Grid& other)
	: width(other.width), height(other.height), density(nullptr), is_wall(nullptr)
{
	int total_cells = width * height;

	density = new double[total_cells];
	std::memcpy(density, other.density, total_cells * sizeof(double));

	velocity_x = new double[total_cells];
	std::memcpy(velocity_x, other.velocity_x, total_cells * sizeof(double));	
	
	velocity_y = new double[total_cells];
	std::memcpy(velocity_y, other.velocity_y, total_cells * sizeof(double));

	is_wall = new bool[total_cells];
	std::memcpy(is_wall, other.is_wall, total_cells * sizeof(bool));

	for (int i = 0; i < direction_num; ++i)
	{
		f_in[i] = new double[total_cells];
		std::memcpy(f_in[i], other.f_in[i], total_cells * sizeof(double));

		f_buffer[i] = new double[total_cells];
		std::memcpy(f_buffer[i], other.f_buffer[i], total_cells * sizeof(double));
	}
}

Grid::~Grid()
{
	delete[] density;
	delete[] velocity_x;
	delete[] velocity_y;
	delete[] is_wall;

	for (int i = 0; i < direction_num; ++i)
	{
		delete[] f_in[i];
		delete[] f_buffer[i];
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
	// Sett cell's properties
	velocity_x[cell_id] = 0.f;
	velocity_y[cell_id] = 0.f;
	is_wall[cell_id] = false;

	density[cell_id] = 1.f;

	// Intialize f_in as equlibrium function
	for (int j = 0; j < direction_num; j++) 
	{
		double ci_dot_u = directions_x[j] * velocity_x[cell_id] +
			directions_y[j] * velocity_y[cell_id];

		double u_square = velocity_x[cell_id] * velocity_x[cell_id] +
			velocity_y[cell_id] * velocity_y[cell_id];

		// Equlibrium function
		double f_eq = weights[j] * density[cell_id] *
			(1.0 + 3.0 * ci_dot_u + 4.5 * ci_dot_u * ci_dot_u - 1.5 * u_square);

		// Initialize f_in as equlibrium function
		f_in[j][cell_id] = f_eq;
			
		// Clear the function buffer as preparation for update
		f_buffer[j][cell_id] = 0.f;
	}
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
	density[cell_id] = 0.f;
	velocity_x[cell_id] = 0.f;
	velocity_y[cell_id] = 0.f;
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
	density[cell_id] = 0.f;
	velocity_x[cell_id] = 0.f;
	velocity_y[cell_id] = 0.f;
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
	return density[cell_id];
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
