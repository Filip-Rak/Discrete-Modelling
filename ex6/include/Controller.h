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

	// Update speed
	const float UPDATE_SPEED_MAX = 300;
	const float UPDATE_SPEED_MIN = 1;
	const float STANDARD_SPEED_CHANGE = 10;
	const float HIGH_SPEED_CHANGE = 50;
	const float LOW_SPEED_CHANGE = 1;
	float update_speed = 100;

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
	void change_update_speed(float change);
};