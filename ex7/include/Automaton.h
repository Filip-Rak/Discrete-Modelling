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
public:
	/* Static & Constants*/
	static const int direction_num = 4;

	/* Structs */
	struct Grid 
	{
		// Dimensions
		int width;
		int height;

		// Arrays for cell data
		double* concentration;	// 0.0 - 1.0
		bool* is_wall;			// Treated as impassable by gas

		// Indexing: [direction][cell_num]
		double* f_in[direction_num];	// Input functions
		double* f_eq[direction_num];	// Equlibrium distibution functions
		double* f_out[direction_num];	// Output functions
	};


private:
	/* Attributes */

	// Components
	AutomatonCUDA cuda_helper;
	Grid grid;

public:
	/* Constructor & Destructor */
	Automaton(int width, int height);
	~Automaton();

	/* Public Methods */
	void generate_random();
	void reset();
};
