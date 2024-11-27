#include "Controller.h"

// Public Methods
// --------------------

// Constructor
Controller::Controller(int window_width, int window_height, int grid_width, int grid_height):
	visualization(window_width, window_height, grid_width, grid_height),
	ui(visualization.get_gui())
{
	initialize_ui();
}

// Main loop
void Controller::run()
{
	while (visualization.is_window_open())
	{
		process_events();

		if (!paused)
		{
			update();
		}

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

}

void Controller::render()
{
	// Clean the window
	visualization.clear();

	// Draw elements
	visualization.draw_grid();
	visualization.draw_ui();

	// Display drawn elements
	visualization.display();
}

// Set callbacks
void Controller::initialize_ui()
{
	// Pass UI-specific dimensions
	float ui_offset_x = visualization.get_ui_view_offset();
	float ui_width = visualization.get_ui_view_width();

	// Create the UI
	ui.initialize(ui_offset_x, ui_width);

	// Register callbacks
	auto pause_button = ui.get_widget_as<tgui::Button>("pause");
	auto reset_button = ui.get_widget_as<tgui::Button>("reset");
	auto slower_button = ui.get_widget_as<tgui::Button>("slower");
	auto faster_button = ui.get_widget_as<tgui::Button>("faster");

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
			pause_button->setText("Start");
		});	
	
	slower_button->onPress([this]()
		{
			std::cout << "Slower pressed\n";
		});	
	
	faster_button->onPress([this]()
		{
			std::cout << "Faster pressed\n";
		});
}