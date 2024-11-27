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

	ui.initialize(ui_offset_x, ui_width);

	// Register callbacks
	ui.get_button("pause")->onPress([this]()
		{
			// Debug
			std::cout << ui.get_button("pause")->getText() << " pressed\n";

			// Toggle the paused state
			paused = !paused;

			// Change the name based on state
			if (paused)
				ui.get_button("pause")->setText("Pause");
			else
				ui.get_button("pause")->setText("Resume");
		});

	ui.get_button("reset")->onPress([this]()
		{
			std::cout << "Reset pressed\n";
			paused = true;
			ui.get_button("pause")->setText("Start");
		});	
	
	ui.get_button("slower")->onPress([this]()
		{
			std::cout << "Slower pressed\n";
		});	
	
	ui.get_button("faster")->onPress([this]()
		{
			std::cout << "Faster pressed\n";
		});
}