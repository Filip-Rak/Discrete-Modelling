#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "AutomatonCUDA.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Grid.h"

class Automaton
{
private:
	/* Attributes */
	int width, height;

	// Components
	AutomatonCUDA cuda_helper;
	Grid grid;
	Grid grid_fallback;

public:
	/* Constructor & Destructor */
	Automaton(int width, int height);
	~Automaton();

	/* Public Methods */
	void generate_random(double probability = 1.f);
	void reset();

	/* Getters */
	Grid* get_grid();
};
