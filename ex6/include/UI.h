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
	std::unordered_map<std::string, tgui::Button::Ptr> buttons;

public:
	/* Constructor */
	explicit UI(tgui::Gui& gui);

	/* Public Methods */

	// Initialize UI elements
	void initialize(float ui_offset_x, float ui_width);

	/* Getters */
	tgui::Button::Ptr UI::get_button(const std::string& name);
};