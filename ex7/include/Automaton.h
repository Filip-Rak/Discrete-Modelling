#pragma once

#include <cstdint>
#include <cstdlib>
#include <map>
#include <bitset>
#include <iostream>

#include "AutomatonCUDA.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Automaton
{
private:
	/* Attributes */

	// Components
	AutomatonCUDA cuda_helper;

	int width;
	int height;

public:
	/* Constructor & Destructor */
	Automaton(int width, int height);
	~Automaton();

	/* Public Methods */
	void generate_random();
	void reset();
};
