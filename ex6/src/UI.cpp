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
	pause_button->onPress([this]()
		{
			if (on_pause) 
				on_pause();
		});

	gui_ref.add(pause_button);

	// Reset Button
	auto reset_button = tgui::Button::create("Reset");
	reset_button->setSize(basic_width, basic_height);
	reset_button->setTextSize(basic_text_size);
	reset_button->setPosition(total_x_offset, top_margin * 2 + basic_height * 1);
	reset_button->onPress([this]()
		{
			if (on_reset)
				on_reset();
		});

	gui_ref.add(reset_button);
}

/* Setters */
void UI::set_pause_callback(const std::function<void()>& callback)
{
	on_pause = callback;
}

void UI::set_reset_callback(const std::function<void()>& callback)
{
	on_reset = callback;
}
