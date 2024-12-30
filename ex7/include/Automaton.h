#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <functional>

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

	std::function<void(int, int)> boundary_condition_function = [this](int x, int y) { this->apply_bc2(x, y); };
	// std::function<void(int, int)> boundary_condition_function = [](int x, int y) { return; };

public:
	/* Constructor & Destructor */
	Automaton(int width, int height);
	~Automaton();

	/* Public Methods */
	void generate_random(double probability = 1.f);
	void reset();
	void update(bool use_gpu);

private:
	/* Private Methods */
	void update_cpu();
	void apply_bc1(int x, int y);
	void apply_bc2(int x, int y);
	void update_gpu();

public:
	/* Getters */
	Grid* get_grid();
};
