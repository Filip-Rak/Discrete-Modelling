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
}

// Main loop
void Controller::run()
{
	while (visualization.is_window_open())
	{
		process_events();
		update();
		render();
	}
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
			std::cout << "Automaton update\n";
		}

	}
}

void Controller::render()
{
	// Clean the window
	visualization.clear();

	// Draw elements
	visualization.draw_grid(automaton.get_cells());
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

	// Get buttons
	auto pause_button = ui.get_widget_as<tgui::Button>("pause");
	auto reset_button = ui.get_widget_as<tgui::Button>("reset");
	auto slower_button = ui.get_widget_as<tgui::Button>("slower");
	auto faster_button = ui.get_widget_as<tgui::Button>("faster");

	// Register callbacks
	pause_button->onPress([this, pause_button]()
		{
			// Debug
			std::cout << pause_button->getText() << " pressed\n";

			// Toggle the paused state
			paused = !paused;

			// Change the name based on state
			if (paused)
				pause_button->setText("Pause");
			else
				pause_button->setText("Resume");
		});

	reset_button->onPress([this, pause_button]()
		{
			std::cout << "Reset pressed\n";
			paused = true;
			automaton.generate_random();
			pause_button->setText("Start");
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
