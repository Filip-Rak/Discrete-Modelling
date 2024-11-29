#pragma once

#include "Visualization.h"
#include "Automaton.h"
#include "UI.h"
// include automaton

#include <iostream>

class Controller 
{
private:
	// Components
	Visualization visualization;
	UI ui;
	Automaton automaton;

	// Flags
	bool use_gpu = false;
	bool paused = true;
	bool text_input_in_use = false;
	Automaton::State selected_state = Automaton::GAS;

	// Update speed
	const float UPDATE_SPEED_MAX = 300;
	const float UPDATE_SPEED_MIN = 1;
	const float STANDARD_SPEED_CHANGE = 10;
	const float HIGH_SPEED_CHANGE = 50;
	const float LOW_SPEED_CHANGE = 1;
	float update_speed = 50;

	// Time
	sf::Clock delta_clock;
	float delta_time;

	float time_between_updates = 1 / update_speed;
	float time_since_update = time_between_updates;

	// Generation
	float MIN_PROBABILITY = 0.f;
	float MAX_PROBABILITY = 1.f;
	float probability = (MIN_PROBABILITY + MAX_PROBABILITY) / 2.f;

public:
	// Constructor
	Controller(int window_width, int window_height, int grid_width, int grid_height);

	// Main loop
	void run();
	void update_clicked_cell(int cell_x, int cell_y);

private:
	void process_events();
	void update();
	void render();
	void initialize_ui();
	void change_update_speed(float direction);
};