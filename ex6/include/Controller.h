#pragma once

#include "Visualization.h"
#include "UI.h"
// include automaton

#include <iostream>

class Controller 
{
private:
	// Components
	Visualization visualization;
	UI ui;
	// !-- automaton

	// Flags
	bool use_gpu = false;
	bool paused = true;

public:
	// Constructor
	Controller(int window_width, int window_height, int grid_width, int grid_height);

	// Main loop
	void run();

private:
	void process_events();
	void update();
	void render();
	void initialize_ui();
};