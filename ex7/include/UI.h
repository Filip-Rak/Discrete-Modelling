#pragma once

#include <TGUI/Backend/SFML-Graphics.hpp>
#include <TGUI/TGUI.hpp>
#include <functional>
#include <string>

class UI
{
private:
	/* Attributes */

	// Components
	tgui::Gui& gui_ref;	// Reference to the GUI within Visualization Class
	std::unordered_map<std::string, tgui::Widget::Ptr> widgets;

public:
	/* Constructor */
	explicit UI(tgui::Gui& gui);

	/* Public Methods */

	// Initialize UI elements
	void initialize(float ui_offset_x, float ui_width, float ctrl_speed, float shift_speed);

	/* Getters */
	tgui::Widget::Ptr get_widget(const std::string& name);

	// Template helper to retrieve a specific widget type
	template<typename T>
	std::shared_ptr<T> get_widget_as(const std::string& name)
	{
		return std::dynamic_pointer_cast<T>(get_widget(name));
	}

	/* Setters */
	void set_speed_label_speed(float speed);	
	void set_fps_label_fps(int fps);
	void set_iteration_label_num(int iteration);
};