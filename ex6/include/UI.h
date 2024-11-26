#pragma once

#include <TGUI/Backend/SFML-Graphics.hpp>
#include <TGUI/TGUI.hpp>
#include <functional>

class UI
{
private:
	/* Attributes */

	// Components
	tgui::Gui& gui_ref;	// Reference to the GUI within Visualization Class

	// Callback functions
	std::function<void()> on_pause;
	std::function<void()> on_reset;

public:
	/* Constructor */
	explicit UI(tgui::Gui& gui);

	/* Public Methods */

	// Initialize UI elements
	void initialize(float ui_offset_x, float ui_width);

	/* Setters */
	void set_pause_callback(const std::function<void()>& callback);
	void set_reset_callback(const std::function<void()>& callback);
};