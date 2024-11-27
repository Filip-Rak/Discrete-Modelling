#include "UI.h"
#include <iostream>

/* Constructor */
UI::UI(tgui::Gui& gui) : gui_ref(gui) {}

/* Public Methods */
void UI::initialize(float ui_offset_x, float ui_width)
{
	float basic_margin = ui_width * 0.15;
	float basic_width = ui_width - basic_margin * 2;
	float total_x_offset = ui_offset_x + basic_margin;
	float basic_height = 60;
	float basic_text_size = 40;
	float top_margin = 20;

	// Pause Button
	auto pause_button = tgui::Button::create("Start");
	pause_button->setSize(basic_width, basic_height);
	pause_button->setTextSize(basic_text_size);
	pause_button->setPosition(total_x_offset, top_margin);
	buttons["pause"] = pause_button;

	gui_ref.add(pause_button);

	// Reset Button
	auto reset_button = tgui::Button::create("Reset");
	reset_button->setSize(basic_width, basic_height);
	reset_button->setTextSize(basic_text_size);
	reset_button->setPosition(total_x_offset, top_margin * 2 + basic_height * 1);
	buttons["reset"] = reset_button;


	gui_ref.add(reset_button);

	// Slower Button
	float left_button_width = (basic_width - basic_margin) / 2;

	auto slower_button = tgui::Button::create("<");
	slower_button->setSize(left_button_width, basic_height);
	slower_button->setTextSize(basic_text_size);
	slower_button->setPosition(total_x_offset, top_margin * 3 + basic_height * 2);
	buttons["slower"] = slower_button;


	gui_ref.add(slower_button);	
	
	// Faster Button
	float total_right_button_offset = total_x_offset + left_button_width + basic_margin;

	auto faster_button = tgui::Button::create(">");
	faster_button->setSize(left_button_width, basic_height);
	faster_button->setTextSize(basic_text_size);
	faster_button->setPosition(total_right_button_offset, top_margin * 3 + basic_height * 2);
	buttons["faster"] = faster_button;

	gui_ref.add(faster_button);
}

/* Getters */
tgui::Button::Ptr UI::get_button(const std::string& name)
{
	return buttons[name];
}
