#pragma once

#include "Visualization.h"
// include ui
// include automaton

class Controller 
{
private:
	// Components
	Visualization visualization;
	// !-- ui
	// !-- automaton

	// Flags
	bool use_gpu = false;
	bool paused = false;

public:
	// Constructor
	Controller(int window_width, int window_height, int grid_width, int grid_height);

	// Main loop
	void run();

private:
	void process_events();

	void update();

	void render();
};