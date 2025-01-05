#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
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

	// ComponentsS
	AutomatonCUDA cuda_helper;
	Grid grid;
	Grid grid_fallback;

	// std::function<void(int, int)> apply_boundry_condition = [this](int x, int y) { this->apply_bc2(x, y); };
	std::function<void(int, int)> apply_boundry_condition = [](int x, int y) { return; };

public:
	/* Constructor & Destructor */
	Automaton(int width, int height);
	~Automaton();

	/* Public Methods */
	void generate_random(double probability = 1.f);
	void reset();
	void update(bool use_gpu);
	void update_particles(double cell_size);
	void save_to_file(std::string path, int iteration);
	int load_from_file(std::string path);

private:
	/* Private Methods */
	void update_cpu();
	void apply_bc1(int x, int y);
	void apply_bc2(int x, int y);
	void update_gpu();
	double clamp(double val, double min, double max);

public:
	/* Getters */
	Grid* get_grid();
};
