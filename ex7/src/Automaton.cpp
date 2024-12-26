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
	// Debug Print the total amount of gas in the system
	double sum_in = 0;
	double sum_c = 0;
	for (int i = 0; i < grid.width * grid.height; i++)
	{
		for (int j = 0; j < 4; j++)
			sum_in += grid.f_in[j][i];

		sum_c += grid.density[i];
	}

	// std::cout << "Total gas in the system:\n\tC: " << sum_c << "\n\tIn: " << sum_in << "\n";

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

			// Loop over all directions
			for (int direction = 0; direction < grid.direction_num; direction++)
			{
				// 1. Collision
				// Compute dot product of velocity and direction vector
				double ci_dot_u = grid.directions_x[direction] * grid.velocity_x[cell_id] +
					grid.directions_y[direction] * grid.velocity_y[cell_id];

				// Compute square of velocity magnitude
				double u_square = grid.velocity_x[cell_id] * grid.velocity_x[cell_id] +
					grid.velocity_y[cell_id] * grid.velocity_y[cell_id];

				// Limit the speed to avoid numerical errors
				if (u_square > 0.1f) 
				{
					double scale = 0.1 / sqrt(u_square);
					grid.velocity_x[cell_id] *= scale;
					grid.velocity_y[cell_id] *= scale;
					u_square = 0.1 * 0.1;
					ci_dot_u = grid.directions_x[direction] * grid.velocity_x[cell_id] +
						grid.directions_y[direction] * grid.velocity_y[cell_id];
				}

				// Equlibrium function
				double f_eq = grid.weights[direction] * grid.density[cell_id] *
					(1.0 + 3.0 * ci_dot_u + 4.5 * ci_dot_u * ci_dot_u - 1.5 * u_square);

				// Output function
				double f_out = grid.f_in[direction][cell_id] +
					(1.0 / grid.tau) * (f_eq - grid.f_in[direction][cell_id]);

				// 2. Streaming
				// Find the neighbour position
				int offset_x = grid.directions_x[direction];
				int offset_y = grid.directions_y[direction];

				int neighbour_x = x + offset_x;
				int neighbour_y = y + offset_y;

				// Check if the neighbour is either out of bounds or a wall
				int neighbour_id = grid.get_id(neighbour_x, neighbour_y);
				if (neighbour_x < 0 || neighbour_x >= grid.width ||
					neighbour_y < 0 || neighbour_y >= grid.height ||
					grid.is_wall[neighbour_id])

				{	// Bounce Back
					int opposite_dir = grid.opposite_directions[direction];
					grid.f_buffer[opposite_dir][cell_id] = f_out;
				}
				else // Within bounds
				{
					grid.f_buffer[direction][neighbour_id] = f_out;
				}
			}
		}
	}

	// Swap the buffer with f_in
	std::swap(grid.f_in, grid.f_buffer);

	// Update prperties of each cell and zero the buffer
	for (int i = 0; i < width * height; i++)
	{
		if (grid.is_wall[i])
		{
			grid.density[i] = 0.f;
			grid.velocity_x[i] = 0.f;
			grid.velocity_y[i] = 0.f;
			continue;
		}

		// Get the sum of all input directions
		double density = 0.f;
		double momentum_x = 0.f;
		double momentum_y = 0.f;

		for (int dir = 0; dir < Grid::direction_num; dir++)
		{
			double input_val = grid.f_in[dir][i];

			// Accumulate density
			density += input_val;

			// Accumulate momentum components
			momentum_x += input_val * grid.directions_x[dir];
			momentum_y += input_val * grid.directions_y[dir];

			// Zero the buffer
			grid.f_buffer[dir][i] = 0.f;
		}

		// Set the density of this cell
		grid.density[i] = density;

		// Avoid division by 0 
		if (density > 0.f)
		{
			grid.velocity_x[i] = momentum_x / density;
			grid.velocity_y[i] = momentum_y / density;
		}
		else
		{
			grid.velocity_x[i] = 0.f;
			grid.velocity_y[i] = 0.f;
		}
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