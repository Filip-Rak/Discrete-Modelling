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

	// Overwrite for now
	wall_start = -1;
	wall_end = -1;
	gas_end = this->width;

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

	// Overwrite boundries
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			if (x == 0 || x == width - 1 || y == 0 || y == height - 1)
			{
				apply_boundry_condition(x, y);
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
				if (u_square > 0.1f && false) 
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

	// Apply boundry conditions
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
			apply_boundry_condition(x, y);
	}

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

void Automaton::apply_bc1(int x, int y)
{
	bool top = (y == 0);
	bool bottom = (y == height - 1);
	bool left = (x == 0);
	bool right = (x == width - 1);

	// Avoid inner cells
	if (!top && !bottom && !left && !right)
	{
		return;
	}

	int cell_id = grid.get_id(x, y);

	/* Apply to every boundry */

	// 1. Zero Y-axis velocity
	grid.velocity_y[cell_id] = 0.f;

	/* Apply to specific boundry */
	double max = 0.02f;
	double min = 0.f;

	if (top)
	{
		grid.velocity_x[cell_id] = max;
	}

	else if (bottom)
	{
		grid.velocity_x[cell_id] = min;
	}

	else if (left || right)
	{
		double multi = (double)y / (double)(grid.height - 1);
		multi = 1 - multi;

		grid.velocity_x[cell_id] = min + (max - min) * multi;
	}

	// Update input functions
	double u_square = grid.velocity_x[cell_id] * grid.velocity_x[cell_id] +
		grid.velocity_y[cell_id] * grid.velocity_y[cell_id];

	for (int direction = 0; direction < Grid::direction_num; direction++)
	{
		double ci_dot_u = grid.directions_x[direction] * grid.velocity_x[cell_id] +
			grid.directions_y[direction] * grid.velocity_y[cell_id];

		grid.f_in[direction][cell_id] = grid.weights[direction] * grid.density[cell_id] *
			(1.0 + 3.0 * ci_dot_u + 4.5 * ci_dot_u * ci_dot_u - 1.5 * u_square);
	}
}

void Automaton::apply_bc2(int x, int y)
{
	bool top = (y == 0);
	bool bottom = (y == height - 1);
	bool left = (x == 0);
	bool right = (x == width - 1);

	// Return for inner cells
	if (!top && !bottom && !left && !right)
	{
		return;
	}

	int cell_id = grid.get_id(x, y);

	/* Bottom Boundary - Bounce Back */
	if (bottom)
	{
		for (int dir = 0; dir < Grid::direction_num; dir++)
		{
			int opp = grid.opposite_directions[dir];
			grid.f_in[opp][cell_id] = grid.f_in[dir][cell_id];
		}
	}

	/* Left Boundary - Open with Applied Speed */
	else if (left)
	{
		// Linear speed change on Vertical Axis from top (y = 0) to bottom (y = height - 1)

		// Normalize
		double normalized = (double)y / (double)(height - 1);
		normalized = 1.0 - normalized;

		// Values to apply
		double Ux = 0.02f * normalized;
		double Uy = 0.f;
		double rho = 1.f;  // Assume density of 1 at the inflow
		
		// Apply
		grid.density[cell_id] = rho;
		grid.velocity_x[cell_id] = Ux;
		grid.velocity_y[cell_id] = Uy;
	}

	/* Top Boundary - Symmetric */
	else if (top)
	{
		grid.f_in[4][cell_id] = grid.f_in[3][cell_id];	// 4a
		grid.f_in[8][cell_id] = grid.f_in[5][cell_id];	// 5a
		grid.f_in[7][cell_id] = grid.f_in[6][cell_id];	// 6a

		grid.velocity_y[cell_id] = 0.0;
	}

	/* Right Boundry - Open with Applied Density = 1.0 */
	else if (right)
	{
		// Apply density
		grid.density[cell_id] = 1.0;

		// 12a (modified)
		double u_x = (grid.f_in[0][cell_id] + grid.f_in[3][cell_id] + grid.f_in[4][cell_id] 
			+ 2 * (grid.f_in[1][cell_id] + grid.f_in[5][cell_id] + grid.f_in[8][cell_id]));
		u_x /= grid.density[cell_id];
		u_x -= 1;

		grid.velocity_x[cell_id] = u_x;
		grid.velocity_y[cell_id] = 0.f;
	}

	/* Update Input Functions */
	double u_square = grid.velocity_x[cell_id] * grid.velocity_x[cell_id] +
		grid.velocity_y[cell_id] * grid.velocity_y[cell_id];

	for (int direction = 0; direction < Grid::direction_num; direction++)
	{
		double ci_dot_u = grid.directions_x[direction] * grid.velocity_x[cell_id] +
			grid.directions_y[direction] * grid.velocity_y[cell_id];

		grid.f_in[direction][cell_id] = grid.weights[direction] * grid.density[cell_id] *
			(1.0 + 3.0 * ci_dot_u + 4.5 * ci_dot_u * ci_dot_u - 1.5 * u_square);
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