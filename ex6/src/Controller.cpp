#include "Controller.h"

// Public Methods
// --------------------

// Constructor
Controller::Controller(int window_width, int window_height, int grid_width, int grid_height):
	visualization(window_width, window_height, grid_width, grid_height),
	ui(visualization.get_gui()),
	automaton(grid_width, grid_height)
{
	initialize_ui();
	automaton.generate_random();

	visualization.set_cell_click_callback([this](int cell_x, int cell_y)
		{
			update_clicked_cell(cell_x, cell_y);
		});
}

// Main loop
void Controller::run()
{
	visualization.update_grid(automaton.get_cells());
	while (visualization.is_window_open())
	{
		process_events();
		update();
		render();
	}
}

void Controller::update_clicked_cell(int cell_x, int cell_y)
{
	std::cout << "update_clicked_cell() called\n";

	// Get the cell from automaton
	auto cells = automaton.get_cells();
	int cell_index = cell_y * automaton.get_width() + cell_x;

	// Update the cell
	auto old_cell = cells[cell_index];
	auto new_cell = Automaton::set_state(old_cell, selected_state);

	// Return the updated cell back into the grid
	cells[cell_index] = new_cell;

	// Update the cell within the grid
	visualization.update_grid_cell(automaton.get_cells(), cell_x, cell_y);
}

// Private Methods
// --------------------
void Controller::process_events()
{
	// Updatet the window
	visualization.process_window_events();

	// Update UI
	// !-- ui.update()

	// Handle UI events
	// !--
}

void Controller::update()
{
	// Delta time update
	delta_time = delta_clock.restart().asSeconds();

	// Fps update
	accumulated_time += delta_time;
	frames_since_update += 1;

	if (accumulated_time >= FPS_UPDATE_INTERVAL)
	{
		// Update the display
		float fps = frames_since_update / accumulated_time;
		ui.set_fps_label_fps(ceil(fps));

		// Reset counter
		frames_since_update = 0;
		accumulated_time = 0;
	}

	// Automaton update
	if (!paused)
	{
		// Increase the time if not paused
		time_since_update += delta_time;
		if (time_since_update > time_between_updates)
		{
			// Reset the time
			time_since_update -= time_between_updates;

			// Update the automaton
			automaton.update();

			// Update the visualization after grid chnages
			visualization.update_grid(automaton.get_cells());
		}

	}
}

void Controller::render()
{
	// Clean the window
	visualization.clear();

	// Draw elements
	visualization.draw_grid(outline_enabled);
	visualization.draw_ui();

	// Display drawn elements
	visualization.display();
}

void Controller::initialize_ui()
{
	// Pass UI-specific dimensions
	float ui_offset_x = visualization.get_ui_view_offset();
	float ui_width = visualization.get_ui_view_width();

	// Create the UI
	ui.initialize(ui_offset_x, ui_width, LOW_SPEED_CHANGE, HIGH_SPEED_CHANGE);
	ui.set_speed_label_speed(update_speed);
	ui.set_fps_label_fps(ceil(1 / delta_time));

	// Get buttons
	auto pause_button = ui.get_widget_as<tgui::Button>("pause");
	auto reset_button = ui.get_widget_as<tgui::Button>("reset");
	auto generate_button = ui.get_widget_as<tgui::Button>("generate");
	auto slower_button = ui.get_widget_as<tgui::Button>("slower");
	auto faster_button = ui.get_widget_as<tgui::Button>("faster");
	auto outline_button = ui.get_widget_as<tgui::Button>("outline");
	auto air_button = ui.get_widget_as<tgui::Button>("air_button");
	auto gas_button = ui.get_widget_as<tgui::Button>("gas_button");
	auto wall_button = ui.get_widget_as<tgui::Button>("wall_button");
	auto probability_text_area = ui.get_widget_as<tgui::TextArea>("prob_input");

	// Register callbacks

	// Control buttons
	pause_button->onPress([this, pause_button]()
		{
			// Debug
			std::cout << pause_button->getText() << " pressed\n";

			// Toggle the paused state
			paused = !paused;

			// Change the name based on state
			if (paused)
				pause_button->setText("Resume");
			else
				pause_button->setText("Pause");
		});

	reset_button->onPress([this, pause_button]()
		{
			std::cout << "Reset pressed\n";
			paused = true;
			automaton.reset();
			pause_button->setText("Start");
			visualization.update_grid(automaton.get_cells());
		});		
	
	generate_button->onPress([this, pause_button, probability_text_area]()
		{
			// Regenrate cells
			float probability_value = std::stof(probability_text_area->getText().toStdString()) / 100.f;
			automaton.generate_random(probability_value);
			visualization.update_grid(automaton.get_cells());

			// Pause
			paused = true;
			pause_button->setText("Start");

			// Debug
			std::cout << "Generate pressed: " << probability_value << "\n";
		});	
	
	slower_button->onPress([this]()
		{
			std::cout << "Slower pressed\n";
			change_update_speed(-1);
		});	
	
	faster_button->onPress([this]()
		{
			std::cout << "Faster pressed\n";
			change_update_speed(1);
		});

	outline_button->onPress([this, outline_button]()
		{
			outline_enabled = !outline_enabled;

			if (outline_enabled)
				outline_button->setText("Show Grid");
			else
				outline_button->setText("Hide Grid");
		});

	// State tools
	air_button->onPress([this]
		{
			selected_state = Automaton::State::EMPTY;
			std::cout << "Select: Air\n";
		});	

	gas_button->onPress([this]
		{
			selected_state = Automaton::State::GAS;
			std::cout << "Select: Gas\n";
		});	

	wall_button->onPress([this]
		{
			selected_state = Automaton::State::WALL;
			std::cout << "Select: Wall\n";
		});

	probability_text_area->setText(std::to_string((int)(probability * 100.f)));

	probability_text_area->onTextChange([this, probability_text_area]()
		{
			// Prevent recursion
			if (text_input_in_use) return;

			text_input_in_use = true; // Set the guard

			std::string text = probability_text_area->getText().toStdString();

			// Remove non-numeric characters
			text.erase(std::remove_if(text.begin(), text.end(),
				[](char c) { return !std::isdigit(c); }),
				text.end());

			// Convert to integer and clamp
			float min = MIN_PROBABILITY * 100, max = MAX_PROBABILITY * 100;
			int value = text.empty() ? min : std::stoi(text);
			if (value > max) value = max;

			// Update the text to reflect the valid percentage
			probability_text_area->setText(std::to_string(value));

			text_input_in_use = false; // Clear the guard
		});
}

void Controller::change_update_speed(float direction)
{
	// Apply speed change value based on keyboard
	float change = STANDARD_SPEED_CHANGE;
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl))
		change = LOW_SPEED_CHANGE;
	else if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
		change = HIGH_SPEED_CHANGE;

	// Find new speed and clamp it
	float new_speed = update_speed + change * direction;
	new_speed = std::max(UPDATE_SPEED_MIN, std::min(new_speed, UPDATE_SPEED_MAX));

	// Update the speed and time
	update_speed = new_speed;
	time_between_updates = 1 / update_speed;

	// Update labels
	std::cout << "Speed: " << update_speed << "\n";
	ui.set_speed_label_speed(update_speed);
}
