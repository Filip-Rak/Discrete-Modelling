#include "Automaton.h"

/* Constructor & Destructor */
Automaton::Automaton(int width, int height) : 
    cuda_helper(width, height)
{
	this->width = width;
	this->height = height;

	// Dynamically allocate space for cell data
}

Automaton::~Automaton()
{
	// Free dynamic cell data
}

/* Public Methods */
void Automaton::generate_random()
{
}
