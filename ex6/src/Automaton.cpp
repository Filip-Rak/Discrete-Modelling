#include "Automaton.h"

/* Constructor & Destructor */
Automaton::Automaton(int width, int height)
{
	this->width = width;
	this->height = height;

	// Dynamically allocate space for cell data
	this->cells = new uint16_t[width * height];
	this->cells_fallback = new uint16_t[width * height];
}

Automaton::~Automaton()
{
	// Free dynamic cell data
	delete[] this->cells;
	delete[] this->cells_fallback;

    this->cells = nullptr;
    this->cells_fallback = nullptr;
}

/* Public Methods */
void Automaton::generate_random_legacy(float probability)
{
    for (int i = 0; i < this->width * this->height; i++) 
    {
        float rand_val = static_cast<float>(rand()) / RAND_MAX;

        if (rand_val < probability || probability == 1.0f) 
        {
            uint16_t cell = 0;
            cell = set_state(cell, GAS);            // Set state to GAS
            cell = set_input(cell, (1 << Automaton::UP) | (1 << Automaton::DOWN));
            cell = set_output(cell, 0);             // Start with no outputs
            this->cells[i] = cell;
        }
        else 
        {
            uint16_t cell = 0;
            cell = set_state(cell, EMPTY);  // Set state to air
            cell = set_input(cell, 0);      // No inputs
            cell = set_output(cell, 0);     // Start with no outputs
            this->cells[i] = cell;
        }

        // Copy to fallback array
        this->cells_fallback[i] = this->cells[i];
    }
}

void Automaton::generate_random(float probability)
{
    int wall_position = this->width / 6; // Adjust proportion for wall position
    int gas_end = wall_position;        // End of gas region
    int wall_start = wall_position;     // Start of wall region
    int wall_end = wall_start + 1;      // End of wall region (1-cell wide)

    for (int y = 0; y < this->height; ++y)
    {
        for (int x = 0; x < this->width; ++x)
        {
            uint16_t cell = 0;
            int index = y * this->width + x;

            if (x < gas_end) // Gas-filled region
            {
                float rand_val = static_cast<float>(rand()) / RAND_MAX;

                if (rand_val < probability || probability == 1.0f)
                {
                    cell = set_state(cell, GAS);            // Set state to GAS
                    cell = set_input(cell, (1 << Automaton::DOWN));
                    cell = set_output(cell, 0);             // Start with no outputs
                }
                else
                {
                    cell = set_state(cell, EMPTY);          // Set state to EMPTY
                    cell = set_input(cell, 0);              // No inputs
                    cell = set_output(cell, 0);             // No outputs
                }
            }
            else if (x >= wall_start && x < wall_end) // Wall region
            {
                cell = set_state(cell, Automaton::WALL);   // Set state to WALL
                cell = set_input(cell, 0);                // No inputs
                cell = set_output(cell, 0);               // No outputs
            }
            else // Empty region
            {
                cell = set_state(cell, EMPTY);            // Set state to EMPTY
                cell = set_input(cell, 0);                // No inputs
                cell = set_output(cell, 0);               // No outputs
            }

            this->cells[index] = cell;

            // Copy to fallback array
            this->cells_fallback[index] = cell;
        }
    }
}

void Automaton::update()
{
    // 1. Collisiion

    // Masks
    uint8_t up_down_mask = (1 << Automaton::UP) | (1 << Automaton::DOWN);
    uint8_t left_right_mask = (1 << Automaton::LEFT) | (1 << Automaton::RIGHT);

    // Create copy of the grid
    uint16_t* updated_cells = new uint16_t[width * height];

    // Handle all collision for every cell first
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int cell_id = j * width + i;
            uint16_t cell = cells[cell_id];

            // Get the current input
            uint8_t input = get_input(cell);

            // Check if only UP and DOWN are active
            if ((input & up_down_mask) == up_down_mask && (input & ~up_down_mask) == 0)
            {
                // Flip UP and DOWN bits
                input = left_right_mask;
            }
            // Check if only LEFT and RIGHT are active
            else if ((input & left_right_mask) == left_right_mask && (input & ~left_right_mask) == 0)
            {
                // Flip LEFT and RIGHT bits
                input = up_down_mask;
            }

            // Convert input into output
            cell = set_output(cell, input);

            // Clear the input
            cell = set_input(cell, 0);  // Dangerous! I don't know if this will work!

            // Add the cell to the copy array
            updated_cells[cell_id] = cell;
        }
    }

    // 2. Streaming
    uint16_t* streamed_inputs = new uint16_t[width * height];
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int cell_id = j * width + i;
            uint16_t cell = updated_cells[cell_id];

            // Skip walls 
            if (get_state(cell) == WALL)
                continue;

            // Get the current output
            uint8_t output = get_output(cell);

            // Move particles to neighbouring cells
            if (output & (1 << Automaton::UP))
            {
                int neighbor_id = (j > 0) ? (j - 1) * width + i : -1; // Boundary check
                output_to(updated_cells, cell_id, neighbor_id, UP, DOWN);
            }

            if (output & (1 << Automaton::DOWN))
            {
                int neighbor_id = (j < height - 1) ? (j + 1) * width + i : -1; // Boundary check
                output_to(updated_cells, cell_id, neighbor_id, DOWN, UP);
            }

            if (output & (1 << Automaton::LEFT))
            {
                int neighbor_id = (i > 0) ? j * width + (i - 1) : -1; // Boundary check
                output_to(updated_cells, cell_id, neighbor_id, LEFT, RIGHT);
            }

            if (output & (1 << Automaton::RIGHT))
            {
                int neighbor_id = (i < width - 1) ? j * width + (i + 1) : -1; // Boundary check
                output_to(updated_cells, cell_id, neighbor_id, RIGHT, LEFT);
            }
        }
    }

    // Update the states
    for (int i = 0; i < height * width; i++)
    {
        uint16_t cell = updated_cells[i];

        // Check if the cell is a wall
        if (get_state(cell) == WALL)
        {
            continue; // Skip updating walls
        }

        // Check if there are any active inputs or outputs
        uint8_t input = get_input(cell);
        uint8_t output = get_output(cell);

        if (input != 0 || output != 0)
        {
            // Set the cell to GAS if any direction is active
            cell = set_state(cell, GAS);
        }
        else
        {
            // Otherwise, set the cell to EMPTY
            cell = set_state(cell, EMPTY);
        }

        // Update the cell in the grid
        updated_cells[i] = cell;
    }

    // Update the cell array
    std::swap(cells, updated_cells);
    delete[] updated_cells;
}

void Automaton::reset()
{
    // Copy original cells to current cells
    for (int i = 0; i < width * height; i++)
        cells[i] = cells_fallback[i];
}

// Transfer input direction into output within a neighbour cell
void Automaton::output_to(uint16_t* output_arr, int sender_id, int receiver_id, Direction forward_direction, Direction opposite_direction)
{
    if (receiver_id == -1 || get_state(output_arr[receiver_id]) == WALL)
    {
        // Reflect forward to opposite direction
        output_arr[sender_id] = set_input(output_arr[sender_id], get_input(output_arr[sender_id]) | (1 << opposite_direction));
    }
    else
    {
        uint16_t receiver_cell = output_arr[receiver_id];
        output_arr[receiver_id] = set_input(receiver_cell, get_input(receiver_cell) | (1 << forward_direction));
    }
}
