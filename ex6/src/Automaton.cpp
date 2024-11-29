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
void Automaton::generate_random(float probability)
{
    for (int i = 0; i < this->width * this->height; i++)
    {
        float rand_val = static_cast<float>(rand()) / RAND_MAX;
        if (rand_val <= probability) 
        {
            uint16_t cell = 0;
            cell = set_state(cell, GAS);
            cell = set_input(cell, rand() & 0x0F);  // Randomize inputs
            cell = set_output(cell, rand() & 0x0F); // Randomize outputs
            this->cells[i] = cell;
        }
        else 
        {
            this->cells[i] = set_state(0, EMPTY);
        }

        this->cells_fallback[i] = this->cells[i];
    }
}

void Automaton::reset()
{
    // Copy original cells to current cells
    for (int i = 0; i < width * height; i++)
        cells[i] = cells_fallback[i];
}